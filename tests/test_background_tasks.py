import ast
import contextlib
import importlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import background_manager as background
from hook_manager import HookManager
from task_store import TaskStore
from tool_manager import BASE_TOOLS, ToolManager


def bash_block(command="echo done", **arguments):
    return SimpleNamespace(type="tool_use", id="call_bash", name="bash",
                           input={"command": command, **arguments})


class BackgroundManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = background.BackgroundManager()
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_start_returns_while_worker_is_still_running(self):
        entered = threading.Event()
        release = threading.Event()
        threads = []
        real_thread = threading.Thread

        def make_thread(**kwargs):
            thread = real_thread(**kwargs)
            threads.append(thread)
            return thread

        def command(_command):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release worker")
            return "finished", 0

        with patch.object(background.threading, "Thread", side_effect=make_thread), \
                patch.object(background, "_run_bash_process", side_effect=command):
            try:
                task_id = self.manager.start(bash_block())
                self.assertTrue(entered.wait(1))
                self.assertEqual(task_id, "bg_0001")
                self.assertTrue(threads[0].daemon)
                self.assertEqual(self.manager.tasks[task_id]["status"], "running")
                self.assertEqual(self.manager.tasks[task_id]["tool_use_id"], "call_bash")
                self.assertEqual(self.manager.collect(), [])
            finally:
                release.set()
                for thread in threads:
                    thread.join(2)
            self.assertFalse(threads[0].is_alive())
        notifications = self.manager.collect()
        self.assertEqual(len(notifications), 1)
        self.assertIn("<status>completed</status>", notifications[0])
        self.assertIn("finished", notifications[0])
        self.assertEqual(self.manager.collect(), [])
        self.assertEqual(self.manager.tasks, {})
        self.assertEqual(self.manager.results, {})

    def test_multiple_tasks_success_failure_timeout_and_exception(self):
        outcomes = [("x" * 600, 0), ("bad command", 2),
                    ("Error: Timeout (120s)", None), RuntimeError("worker failed")]
        with patch.object(background.threading, "Thread"), \
                patch.object(background, "_run_bash_process", side_effect=outcomes):
            ids = [self.manager.start(bash_block(f"command {i}")) for i in range(4)]
            self.assertEqual(ids, [f"bg_{i:04d}" for i in range(1, 5)])
            for task_id in ids:
                self.manager._run(task_id, self.manager.tasks[task_id]["command"])
        notifications = self.manager.collect()
        self.assertEqual(len(notifications), 4)
        self.assertIn("<summary>" + "x" * 500 + "</summary>", notifications[0])
        self.assertIn("<status>completed</status>", notifications[0])
        for item in notifications[1:]:
            self.assertIn("<status>failed</status>", item)
        self.assertIn("status 2", notifications[1])
        self.assertIn("Timeout (120s)", notifications[2])
        self.assertIn("RuntimeError: worker failed", notifications[3])
        self.assertEqual(self.manager.collect(), [])

    def test_invalid_command_and_non_bash_do_not_create_task(self):
        for command in (None, "", "  ", 123):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.manager.start(bash_block(command))
        with self.assertRaises(ValueError):
            self.manager.start(SimpleNamespace(name="read_file", input={}))
        self.assertEqual(self.manager.tasks, {})
        self.assertEqual(self.manager._counter, 0)

    def test_thread_start_failure_removes_record(self):
        with patch.object(background.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("cannot start")
            with self.assertRaisesRegex(RuntimeError, "cannot start"):
                self.manager.start(bash_block())
        self.assertEqual(self.manager.tasks, {})
        self.assertEqual(self.manager.results, {})

    def test_only_literal_true_for_bash_requests_background(self):
        self.assertTrue(background.should_run_background("bash", {"run_in_background": True}))
        for value in (False, None, 1, "true"):
            self.assertFalse(background.should_run_background("bash", {"run_in_background": value}))
        self.assertFalse(background.should_run_background("bash", {}))
        self.assertFalse(background.should_run_background("read_file", {"run_in_background": True}))

    def test_notification_injection_preserves_existing_messages(self):
        result = {"type": "tool_result", "tool_use_id": "call_bash", "content": "started"}
        for messages in ([], [{"role": "user", "content": "next question"}],
                         [{"role": "user", "content": [result.copy()]}],
                         [{"role": "assistant", "content": "done"}]):
            with self.subTest(messages=messages), \
                    patch.object(background, "BACKGROUND", self.manager), \
                    patch.object(background.threading, "Thread"), \
                    patch.object(background, "_run_bash_process", return_value=("done", 0)):
                task_id = background.start_background_task(bash_block())
                self.manager._run(task_id, "echo done")
                original = [dict(item) for item in messages]
                self.assertEqual(background.inject_background_results(messages), 1)
                blocks = messages[-1]["content"]
                self.assertEqual(messages[-1]["role"], "user")
                self.assertEqual(blocks[-1]["type"], "text")
                self.assertIn(task_id, blocks[-1]["text"])
                if original and original[-1]["role"] == "user":
                    self.assertEqual(len(messages), len(original))
                    if isinstance(original[-1]["content"], str):
                        self.assertEqual(blocks[0], {"type": "text", "text": "next question"})
                    else:
                        self.assertEqual(blocks[0], result)
                elif original:
                    self.assertEqual(messages[0], original[0])
                self.assertEqual(background.inject_background_results(messages), 0)

    def test_transplanted_definitions_match_reference(self):
        root = Path(__file__).resolve().parents[1] / "src"
        reference = ast.parse((root / "code.py").read_text(encoding="utf-8"))
        actual = ast.parse((root / "background_manager.py").read_text(encoding="utf-8"))
        expected = {node.name: node for node in reference.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for node in actual.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertIn(node.name, expected)
                # Preserve the user's pre-existing cleanup functions and comments.
                if node.name not in {"_stop_process_group", "_stop_all_shell_processes",
                                     "_handle_termination_signal"}:
                    self.assertEqual(ast.dump(node), ast.dump(expected[node.name]))


class ShellProcessTests(unittest.TestCase):
    def test_popen_registers_process_and_cleans_up_on_both_platforms(self):
        for platform in ("nt", "posix"):
            with self.subTest(platform=platform):
                process = Mock(returncode=0)

                def communicate(timeout):
                    self.assertEqual(timeout, 120)
                    self.assertIn(process, background._shell_processes)
                    return "  output", " error  "

                process.communicate.side_effect = communicate
                with patch.object(background.os, "name", platform), \
                        patch.object(background.subprocess, "Popen", return_value=process) as popen, \
                        patch.object(background, "_stop_process_group") as stop:
                    self.assertEqual(background._run_bash_process("command"), ("output error", 0))
                kwargs = popen.call_args.kwargs
                self.assertEqual(kwargs["cwd"], background.WORKDIR)
                self.assertTrue(kwargs["shell"])
                self.assertEqual(kwargs.get("start_new_session"), True if platform == "posix" else None)
                stop.assert_called_once_with(process)
                process.wait.assert_called_once_with(timeout=0.2)
                self.assertNotIn(process, background._shell_processes)

    def test_timeout_spawn_error_empty_output_and_output_limit(self):
        for output, expected in (("", "(no output)"), ("x" * 60000, "x" * 50000)):
            process = Mock(returncode=0)
            process.communicate.return_value = (output, "")
            with patch.object(background.subprocess, "Popen", return_value=process), \
                    patch.object(background, "_stop_process_group"):
                self.assertEqual(background._run_bash_process("command"), (expected, 0))
        process = Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired("command", 120)
        process.wait.side_effect = subprocess.TimeoutExpired("command", 0.2)
        with patch.object(background.subprocess, "Popen", return_value=process), \
                patch.object(background, "_stop_process_group") as stop:
            self.assertEqual(background._run_bash_process("command"), ("Error: Timeout (120s)", None))
        stop.assert_called_once_with(process)
        self.assertNotIn(process, background._shell_processes)
        with patch.object(background.subprocess, "Popen", side_effect=OSError("spawn failed")):
            self.assertEqual(background._run_bash_process("command"), ("Error: OSError: spawn failed", None))

    def test_windows_cleanup_tries_terminate_then_kill_and_skips_exited_process(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("command", 0.05), None]
        with patch.object(background.os, "name", "nt"):
            background._stop_process_group(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [call(timeout=0.05), call(timeout=0.05)])
        process = Mock()
        process.poll.return_value = 0
        with patch.object(background.os, "name", "nt"):
            background._stop_process_group(process)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_posix_cleanup_signals_group_and_handles_missing_group(self):
        process = SimpleNamespace(pid=4321)
        with patch.object(background.os, "name", "posix"), \
                patch.object(background.os, "killpg", create=True) as killpg, \
                patch.object(background.time, "sleep") as sleep:
            background._stop_process_group(process)
        self.assertEqual(killpg.call_args_list,
                         [call(4321, signal.SIGTERM), call(4321, getattr(signal, "SIGKILL", signal.SIGTERM))])
        self.assertEqual(sleep.call_args_list, [call(0.05), call(0.05)])
        with patch.object(background.os, "name", "posix"), \
                patch.object(background.os, "killpg", create=True, side_effect=ProcessLookupError) as killpg, \
                patch.object(background.time, "sleep") as sleep:
            background._stop_process_group(process)
        killpg.assert_called_once()
        sleep.assert_not_called()

    def test_termination_handler_cleans_all_registered_processes(self):
        processes = {Mock(), Mock()}
        with patch.object(background, "_shell_processes", processes), \
                patch.object(background, "_stop_process_group") as stop:
            with self.assertRaises(SystemExit) as caught:
                background._handle_termination_signal(signal.SIGTERM, None)
        self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
        self.assertEqual({item.args[0] for item in stop.call_args_list}, processes)

    def test_real_shell_success_failure_and_registration_cleanup(self):
        commands = [("echo background-smoke", "background-smoke", 0),
                    ("exit /b 3" if os.name == "nt" else "exit 3", "(no output)", 3)]
        for command, output, code in commands:
            with self.subTest(command=command):
                self.assertEqual(background._run_bash_process(command), (output, code))
        self.assertEqual(background._shell_processes, set())


class BackgroundToolAndLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.hooks = HookManager(self.root)
        for callbacks in self.hooks.hooks.values():
            callbacks.clear()
        self.pre = Mock(return_value=None)
        self.post = Mock(return_value=None)
        self.hooks.register_hook("PreToolUse", self.pre)
        self.hooks.register_hook("PostToolUse", self.post)
        self.manager = ToolManager(self.root, self.hooks, SimpleNamespace(load=Mock()),
                                   Mock(), TaskStore(self.root, self.root / ".tasks"))
        self.background = background.BackgroundManager()
        patcher = patch.object(background, "BACKGROUND", self.background)
        patcher.start()
        self.addCleanup(patcher.stop)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_schema_and_dispatch_only_enable_main_agent_background(self):
        main_bash = next(tool for tool in self.manager.main_tools if tool["name"] == "bash")
        self.assertEqual(main_bash["input_schema"]["properties"]["run_in_background"], {"type": "boolean"})
        self.assertEqual(main_bash["input_schema"]["required"], ["command"])
        self.assertEqual(self.manager.sub_tools, BASE_TOOLS)
        self.assertNotIn("run_in_background", BASE_TOOLS[0]["input_schema"]["properties"])
        with patch("tool_manager.start_background_task", return_value="bg_0001") as start, \
                patch("tool_manager._run_bash_process", return_value=("done", 0)) as run:
            block = bash_block(run_in_background=True)
            output = self.manager.execute_tool(block, self.manager.main_handlers)
            self.assertIn("Background task bg_0001 started", output)
            start.assert_called_once_with(block)
            run.assert_not_called()
            self.post.assert_called_with(block, output)
            for arguments in ({}, {"run_in_background": False}, {"run_in_background": 1}):
                self.assertEqual(self.manager.execute_tool(bash_block(**arguments), self.manager.main_handlers), "done")
            self.assertEqual(self.manager.execute_tool(block, self.manager.sub_handlers), "done")
            self.assertEqual(start.call_count, 1)
            self.assertEqual(run.call_count, 4)

    def test_pre_hook_and_existing_danger_checks_prevent_background_start(self):
        with patch("tool_manager.start_background_task") as start:
            self.pre.return_value = "Permission denied"
            self.assertEqual(self.manager.execute_tool(bash_block(run_in_background=True), self.manager.main_handlers),
                             "Permission denied")
            self.post.assert_not_called()
            self.pre.return_value = None
            for command in ("rm -rf /", "sudo command", "shutdown", "reboot", "echo x > /dev/example"):
                block = bash_block(command, run_in_background=True)
                output = self.manager.execute_tool(block, self.manager.main_handlers)
                self.assertEqual(output, "Error: Dangerous command blocked")
                self.post.assert_called_with(block, output)
            start.assert_not_called()

    def test_background_validation_and_start_errors_return_tool_results(self):
        for command in (None, "", " "):
            output = self.manager.execute_tool(bash_block(command, run_in_background=True), self.manager.main_handlers)
            self.assertEqual(output, "Error: Bash command cannot be empty")
        with patch.object(background.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("cannot start")
            self.assertEqual(self.manager.execute_tool(bash_block(run_in_background=True), self.manager.main_handlers),
                             "Error: cannot start")
        self.assertEqual(self.background.tasks, {})
        with patch("tool_manager._run_bash_process", return_value=("failure", 3)):
            self.assertEqual(self.manager.run_bash("command"), "Error: command exited with status 3\nfailure")

    def test_loop_delivers_notification_after_prepare_and_keeps_tool_result(self):
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock()))
        with patch("anthropic.Anthropic", return_value=client), \
                patch("dotenv.load_dotenv"), patch.dict(os.environ, {"MODEL_ID": "test-model"}):
            loop = importlib.import_module("loop_agent")
        final = SimpleNamespace(type="text", text="done")
        prepared = []

        def prepare(messages, request):
            prepared.append(str(messages))
            for task_id in list(self.background.tasks):
                self.background._run(task_id, "echo done")
            return messages

        def reply(**kwargs):
            if client.messages.create.call_count == 1:
                return SimpleNamespace(content=[bash_block(run_in_background=True)])
            blocks = kwargs["messages"][-1]["content"]
            self.assertEqual(blocks[0]["type"], "tool_result")
            self.assertIn("bg_0001 started", blocks[0]["content"])
            self.assertIn("<status>completed</status>", blocks[1]["text"])
            self.assertIn("finished command", blocks[1]["text"])
            self.assertIn("independent Bash commands", kwargs["system"])
            return SimpleNamespace(content=[final])

        client.messages.create.side_effect = reply
        memory = SimpleNamespace(load_memories=Mock(return_value=""), augment_system=lambda system, _: system,
                                 extract_memories=Mock(), consolidate_memories=Mock())
        messages = [{"role": "user", "content": "start independent command"}]
        with patch.object(loop, "client", client), patch.object(loop, "TOOL_MANAGER", self.manager), \
                patch.object(loop, "HOOK_MANAGER", self.hooks), patch.object(loop, "MEMORY_STORE", memory), \
                patch.object(loop, "COMPACTOR", SimpleNamespace(prepare=prepare)), \
                patch.object(background.threading, "Thread"), \
                patch.object(background, "_run_bash_process", return_value=("finished command", 0)):
            loop.agent_loop(messages, "start independent command")
        self.assertEqual(client.messages.create.call_count, 2)
        self.assertTrue(all("task_notification" not in item for item in prepared))
        self.assertEqual(self.background.tasks, {})
        memory.extract_memories.assert_called_once_with(messages)

    def test_main_loop_does_not_wait_and_subagent_does_not_collect(self):
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock()))
        with patch("anthropic.Anthropic", return_value=client), \
                patch("dotenv.load_dotenv"), patch.dict(os.environ, {"MODEL_ID": "test-model"}):
            loop = importlib.import_module("loop_agent")
        final = SimpleNamespace(type="text", text="done")
        client.messages.create.side_effect = [SimpleNamespace(content=[bash_block(run_in_background=True)]),
                                              SimpleNamespace(content=[final])]
        memory = SimpleNamespace(load_memories=Mock(return_value=""), augment_system=lambda system, _: system,
                                 extract_memories=Mock(), consolidate_memories=Mock())
        messages = [{"role": "user", "content": "start command"}]
        with patch.object(loop, "client", client), patch.object(loop, "TOOL_MANAGER", self.manager), \
                patch.object(loop, "HOOK_MANAGER", self.hooks), patch.object(loop, "MEMORY_STORE", memory), \
                patch.object(loop, "COMPACTOR", SimpleNamespace(prepare=lambda messages, _: messages)), \
                patch.object(background.threading, "Thread"):
            loop.agent_loop(messages, "start command")
            self.assertEqual(self.background.tasks["bg_0001"]["status"], "running")
            with patch.object(background, "_run_bash_process", return_value=("finished later", 0)):
                self.background._run("bg_0001", "echo done")
            client.messages.create.side_effect = None
            client.messages.create.return_value = SimpleNamespace(content=[final])
            with patch.object(loop, "inject_background_results") as inject:
                self.assertEqual(loop.run_subagent("explore"), "done")
                inject.assert_not_called()
            self.assertIn("bg_0001", self.background.results)
            messages.append({"role": "user", "content": "next request"})
            loop.agent_loop(messages, "next request")
        self.assertIn("finished later", messages[-2]["content"][-1]["text"])
        self.assertEqual(self.background.results, {})


if __name__ == "__main__":
    unittest.main()
