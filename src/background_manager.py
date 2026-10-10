import os
import signal
import time
import subprocess
import threading
import atexit
from pathlib import Path


WORKDIR = Path.cwd()


_shell_processes: set[subprocess.Popen] = set()
_shell_process_lock = threading.RLock()



# 停止已经启动的 shell 进程，并尝试等待它退出
def _stop_process_group(process: subprocess.Popen):
    if os.name == "nt":
        """
        分别尝试terminate()和kill()来终止进程
        poll() 用来检查进程状态 如果结束一般会返回int
        """
        for stop in (process.terminate, process.kill):
            if process.poll() is not None:
                return
            try:
                stop()
            except OSError:
                return
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                continue
        return

    # 适配POSIX系统的杀进程方法
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, OSError):
            return
        time.sleep(0.05)

# 终止所有进程
def _stop_all_shell_processes():
    with _shell_process_lock:
        processes = list(_shell_processes)
    for process in processes:
        _stop_process_group(process)

# 处理终止信号 先终止shell中的所有进程 然后退出shell
def _handle_termination_signal(signum, _frame):
    _stop_all_shell_processes()
    raise SystemExit(128 + signum)      # 一般传递143

# 注册程序退出时的清理函数
atexit.register(_stop_all_shell_processes)
# 注册收到sigterm时的处理函数
signal.signal(signal.SIGTERM, _handle_termination_signal)


def _run_bash_process(command: str) -> tuple[str, int | None]:
    process = None
    try:
        # start_new_session (setsid) exists only on POSIX; skip it on Windows.
        popen_kwargs = {} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, errors="replace",
            **popen_kwargs,
        )
        with _shell_process_lock:
            _shell_processes.add(process)
        stdout, stderr = process.communicate(timeout=120)
        output = (stdout + stderr).strip()
        return (output[:50000] if output else "(no output)"), process.returncode
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)", None
    except OSError as error:
        return f"Error: {type(error).__name__}: {error}", None
    finally:
        if process is not None:
            _stop_process_group(process)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            with _shell_process_lock:
                _shell_processes.discard(process)


def _format_bash_result(output: str, exit_code: int | None) -> str:
    if exit_code in (0, None):
        return output
    return f"Error: command exited with status {exit_code}\n{output}"


# 管理在后台运行的任务
class BackgroundManager:
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.results: dict[str, str] = {}
        self._ready: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()

    def start(self, block) -> str:
        if block.name != "bash":
            raise ValueError("Only Bash commands can run in the background")
        command = block.input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash command cannot be empty")

        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            self.tasks[task_id] = {
                "tool_use_id": block.id,
                "command": command,
                "status": "running",
            }

        thread = threading.Thread(
            target=self._run,
            args=(task_id, command),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
            raise
        print(f"  [background] started {task_id}: {command[:60]}")
        return task_id

    def _run(self, task_id: str, command: str):
        try:
            output, exit_code = _run_bash_process(command)
            result = _format_bash_result(output, exit_code)
            status = "completed" if exit_code == 0 else "failed"
        except Exception as error:
            result = f"Error: {type(error).__name__}: {error}"
            status = "failed"

        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            self.results[task_id] = result
            self._ready.append(task_id)

    def collect(self) -> list[str]:
        with self._lock:
            ready = []
            for task_id in self._ready:
                task = self.tasks.pop(task_id, None)
                result = self.results.pop(task_id, "")
                if task is not None:
                    ready.append((task_id, task, result))
            self._ready.clear()

        notifications = []
        for task_id, task, result in ready:
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{result[:500]}</summary>\n"
                f"</task_notification>"
            )
            print(f"  [background] collected {task_id}: {task['status']}")
        return notifications


BACKGROUND = BackgroundManager()
background_tasks = BACKGROUND.tasks
background_results = BACKGROUND.results


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    return (
        tool_name == "bash"
        and tool_input.get("run_in_background") is True
    )


def start_background_task(block) -> str:
    return BACKGROUND.start(block)


def collect_background_results() -> list[str]:
    return BACKGROUND.collect()


def inject_background_results(messages: list) -> int:
    notifications = collect_background_results()
    if not notifications:
        return 0

    blocks = [{"type": "text", "text": item} for item in notifications]
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            content.extend(blocks)
        else:
            messages[-1]["content"] = [
                {"type": "text", "text": str(content)},
                *blocks,
            ]
    else:
        messages.append({"role": "user", "content": blocks})
    return len(notifications)

