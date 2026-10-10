import ast
import contextlib
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hook_manager import HookManager
from task_store import TASK_ID_PATTERN, TaskStore
from tool_manager import BASE_TOOLS, TASK_TOOLS, ToolManager


class TaskToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.store = TaskStore(self.root, self.root / ".tasks")
        self.hooks = HookManager(self.root)
        for callbacks in self.hooks.hooks.values():
            callbacks.clear()
        self.pre = Mock(return_value=None)
        self.post = Mock(return_value=None)
        self.hooks.register_hook("PreToolUse", self.pre)
        self.hooks.register_hook("PostToolUse", self.post)
        self.delegate = Mock(return_value="Subagent result")
        self.manager = ToolManager(self.root, self.hooks, SimpleNamespace(load=Mock()),
                                   self.delegate, task_store=self.store)

    def execute(self, name, **arguments):
        block = SimpleNamespace(name=name, input=arguments)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.manager.execute_tool(block, self.manager.main_handlers)

    def create(self, subject, **arguments):
        output = self.execute("create_task", subject=subject, **arguments)
        return re.search(r"task_[0-9a-f]{8}", output).group()

    def test_task_tools_only_registered_for_main_agent(self):
        names = {tool["name"] for tool in TASK_TOOLS}
        self.assertEqual(names, {"create_task", "update_task", "list_tasks", "get_task",
                                 "claim_task", "complete_task"})
        self.assertTrue(names.issubset(self.manager.main_handlers))
        self.assertTrue(names.issubset({tool["name"] for tool in self.manager.main_tools}))
        self.assertEqual(self.manager.sub_tools, BASE_TOOLS)
        self.assertEqual(set(self.manager.sub_handlers), {tool["name"] for tool in BASE_TOOLS})
        self.assertNotIn("todo_write", self.manager.main_handlers)
        self.assertNotIn("todo_write", {tool["name"] for tool in self.manager.main_tools})
        self.assertIs(self.manager.task_store, self.store)
        self.assertEqual(self.execute("task", prompt="Explore files"), "Subagent result")
        self.delegate.assert_called_once_with(prompt="Explore files")

    def test_all_six_tools_and_feedback(self):
        self.assertEqual(self.execute("list_tasks"), "No tasks. Use create_task to add some.")
        first = self.create("基础", description="中文描述")
        second = self.create("接口")
        self.assertEqual(self.execute("update_task", task_id=second, addBlockedBy=[first]),
                         f"Updated {second} blockedBy: {first}")
        details = json.loads(self.execute("get_task", task_id=first))
        self.assertEqual(details["description"], "中文描述")
        self.assertEqual(details["status"], "pending")
        listing = self.execute("list_tasks")
        self.assertIn(f"[ ] {second}: 接口 [pending] (blockedBy: {first})", listing)
        self.assertTrue(self.execute("claim_task", task_id=second).startswith("Error: Blocked by:"))
        self.assertEqual(self.execute("claim_task", task_id=first), f"Claimed {first} (基础)")
        self.assertIn(f"[>] {first}: 基础 [in_progress] [agent]", self.execute("list_tasks"))
        self.assertEqual(self.execute("complete_task", task_id=first),
                         f"Completed {first} (基础)\nUnblocked: 接口")
        self.assertIn(f"[x] {first}: 基础 [completed] [agent]", self.execute("list_tasks"))
        self.execute("claim_task", task_id=second)
        self.assertEqual(self.execute("complete_task", task_id=second), f"Completed {second} (接口)")

    def test_hooks_receive_success_and_error_outputs(self):
        task_id = self.create("Work")
        block, = self.pre.call_args.args
        self.assertEqual(block.name, "create_task")
        self.post.assert_called_once_with(block, f"Created {task_id}: Work")
        self.pre.reset_mock()
        self.post.reset_mock()
        output = self.execute("get_task", task_id="../outside")
        self.assertTrue(output.startswith("Error: Invalid task ID:"))
        block, = self.pre.call_args.args
        self.post.assert_called_once_with(block, output)

    def test_pre_hook_can_block_without_creating_records(self):
        self.pre.return_value = "Blocked by hook"
        self.assertEqual(self.execute("create_task", subject="Rejected"), "Blocked by hook")
        self.post.assert_not_called()
        self.assertFalse(self.store.directory.exists())

    def test_invalid_tool_arguments_and_owner_become_errors(self):
        task_id = self.create("Work")
        self.assertTrue(self.execute("create_task", subject="").startswith("Error:"))
        self.assertTrue(self.execute("get_task").startswith("Error:"))
        self.assertTrue(self.execute("update_task", task_id=task_id,
                                     addBlockedBy=["../outside"]).startswith("Error:"))
        self.store.claim(task_id, owner="other")
        before = (self.store.directory / f"{task_id}.json").read_bytes()
        output = self.execute("complete_task", task_id=task_id)
        self.assertIn("owned by other, not agent", output)
        self.assertEqual((self.store.directory / f"{task_id}.json").read_bytes(), before)

    def test_task_tools_match_reference_code(self):
        # 只加载参考文件的任务定义，不运行其客户端初始化或 agent 循环。
        source = Path(__file__).resolve().parents[1] / "src" / "code.py"
        names = {"Task", "TaskStore", "create_task", "update_task", "load_task", "list_tasks",
                 "get_task", "incomplete_dependencies", "can_start", "claim_task", "complete_task",
                 "run_create_task", "run_update_task", "run_list_tasks", "run_get_task",
                 "run_claim_task", "run_complete_task"}
        tree = ast.parse(source.read_text(encoding="utf-8"))
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        namespace = {"json": json, "re": re, "secrets": secrets, "Path": Path,
                     "dataclass": dataclass, "asdict": asdict, "WORKDIR": self.root,
                     "TASK_ID_PATTERN": TASK_ID_PATTERN}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
        namespace["TASKS"] = namespace["TaskStore"](self.root / ".reference_tasks")

        def compare(name, **arguments):
            actual = self.execute(name, **arguments)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = namespace[f"run_{name}"](**arguments)
            # 主项目沿用现有分发器的 Error: 前缀。
            self.assertEqual(actual.removeprefix("Error: "), expected)

        compare("list_tasks")
        for suffix, subject in (("11111111", "基础"), ("22222222", "接口"), ("33333333", "测试")):
            with patch("task_store.secrets.token_hex", return_value=suffix):
                compare("create_task", subject=subject, description="中文描述")
        compare("update_task", task_id="task_22222222", addBlockedBy=["task_11111111"])
        compare("update_task", task_id="task_33333333", addBlockedBy=["task_22222222"])
        compare("get_task", task_id="task_22222222")
        compare("claim_task", task_id="task_22222222")
        for task_id in ("task_11111111", "task_22222222", "task_33333333"):
            compare("claim_task", task_id=task_id)
            compare("list_tasks")
            compare("complete_task", task_id=task_id)
        compare("list_tasks")
        for task_id in ("task_11111111", "task_22222222", "task_33333333"):
            actual = (self.store.directory / f"{task_id}.json").read_bytes()
            expected = (namespace["TASKS"].directory / f"{task_id}.json").read_bytes()
            self.assertEqual(actual, expected)

    def test_parent_loop_uses_injected_store_and_returns_results_to_model(self):
        import anthropic
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock()))
        with patch.object(anthropic, "Anthropic", return_value=client), \
                patch("dotenv.load_dotenv"), patch.dict(os.environ, {"MODEL_ID": "test-model"}):
            import loop_agent
        self.assertIs(loop_agent.TOOL_MANAGER.task_store, loop_agent.TASK_STORE)

        def tool(name, **arguments):
            return SimpleNamespace(type="tool_use", id=f"call_{name}", name=name, input=arguments)

        def reply(**kwargs):
            turn = client.messages.create.call_count
            if turn == 1:
                content = [tool("list_tasks")]
            elif turn == 2:
                content = [tool("create_task", subject="Schema"),
                           SimpleNamespace(type="tool_use", id="call_create_api", name="create_task",
                                           input={"subject": "API"})]
            else:
                ids = {task.subject: task.id for task in self.store.list()}
                if turn == 3:
                    content = [tool("update_task", task_id=ids["API"], addBlockedBy=[ids["Schema"]]),
                               tool("claim_task", task_id=ids["API"])]
                elif turn == 4:
                    content = [tool("claim_task", task_id=ids["Schema"])]
                elif turn == 5:
                    content = [tool("complete_task", task_id=ids["Schema"])]
                elif turn == 6:
                    content = [tool("claim_task", task_id=ids["API"])]
                elif turn == 7:
                    content = [tool("complete_task", task_id=ids["API"])]
                else:
                    content = [SimpleNamespace(type="text", text="Done")]
            self.assertIs(kwargs["tools"], self.manager.main_tools)
            return SimpleNamespace(content=content)

        client.messages.create.side_effect = reply
        memory = SimpleNamespace(load_memories=Mock(return_value=""),
                                 augment_system=lambda system, memories: system,
                                 extract_memories=Mock(), consolidate_memories=Mock())
        compactor = SimpleNamespace(prepare=lambda messages, request: messages)
        messages = [{"role": "user", "content": "Implement the API"}]
        with patch.object(loop_agent, "client", client), \
                patch.object(loop_agent, "TOOL_MANAGER", self.manager), \
                patch.object(loop_agent, "HOOK_MANAGER", self.hooks), \
                patch.object(loop_agent, "MEMORY_STORE", memory), \
                patch.object(loop_agent, "COMPACTOR", compactor), \
                contextlib.redirect_stdout(io.StringIO()):
            loop_agent.agent_loop(messages, "Implement the API")

        self.assertEqual(client.messages.create.call_count, 8)
        self.assertTrue(all(task.status == "completed" and task.owner == "agent"
                            for task in self.store.list()))
        system = client.messages.create.call_args.kwargs["system"]
        self.assertIn("Create all task nodes", system)
        self.assertIn("runtime-generated IDs", system)
        self.assertIn("update_task", system)
        results = [block["content"] for message in messages if message["role"] == "user"
                   and isinstance(message["content"], list) for block in message["content"]]
        self.assertTrue(any(output.startswith("Error: Blocked by:") for output in results))
        self.assertTrue(any("Unblocked: API" in output for output in results))
        self.assertEqual(messages[-1]["content"][0].text, "Done")
        memory.extract_memories.assert_called_once_with(messages)
        memory.consolidate_memories.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
