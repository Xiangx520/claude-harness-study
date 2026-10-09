import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from hook_manager import HookManager
from skill_loader import SkillLoader
from todo_manager import TodoManager


# 工具定义告诉模型可用能力；基础工具仅包括命令执行和文件操作
BASE_TOOLS = [
    {
        "name": "bash",                                     # 工具名
        "description": "Run a shell command.",              # 描述 交给llm判断是否调用tool_block
        "input_schema": {                                   # 参数 json格式 需要llm按要求传递参数
            "type": "object",                               # 参数整体是一个对象 即dict
            "properties": {"command": {"type": "string"}},  # 对象的属性 一个名为command的str
            "required": ["command"],                        # 必要性检查
        },
    },
    {   "name": "read_file", "description": "Read file contents.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {   "name": "write_file", "description": "Write content to a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {   "name": "edit_file", "description": "Replace exact text in a file once.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {   "name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
        "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

# skill加载工具
SKILL_TOOL = {
    "name": "load_skill",
    "description": "Load the full SKILL.md content by skill name.",
    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
    "required": ["name"],
}

# 主 agent 的任务列表工具
TODO_TOOL = {
    "name": "todo_write", "description": "Create and manage a task list for your current coding session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    }
                }
            }
        }
    }
}

# 分发任务给子agent
TASK_TOOL = {
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
    },
}

# 上下文压缩工具
COMPACT_TOOL = {
    "name": "compact",
    "description": "Summarize earlier conversation to free context space.",
    "input_schema": {"type": "object", "properties": {}},
}


class ToolManager:
    """Manage tool definitions, handlers, and execution through agent hooks."""

    def __init__(self, workdir: Path, hook_manager: HookManager,
                 skill_loader: SkillLoader, task_handler: Callable[[str], str]):
        self.workdir = workdir
        self.hook_manager = hook_manager
        self.todo = TodoManager()

        # 子 agent 仅使用基础能力，复制容器以便独立组装。
        self.sub_tools = list(BASE_TOOLS)
        self.sub_handlers = {
            "bash": self.run_bash,
            "read_file": self.run_read,
            "write_file": self.run_write,
            "edit_file": self.run_edit,
            "glob": self.run_glob,
        }
        self.main_tools = [*BASE_TOOLS, SKILL_TOOL, TODO_TOOL, TASK_TOOL, COMPACT_TOOL]
        # compact 由主循环在工具批次结束后处理，不进入普通分发器。
        self.main_handlers = {
            **self.sub_handlers,
            "load_skill": skill_loader.load,
            "todo_write": self.run_todo_write,
            "task": task_handler,
        }

    # 执行command并返回结果给调用方
    def run_bash(self, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        # 危险操作检查
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        try:
            # 运行bash
            r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                               capture_output=True, text=True, errors="replace", timeout=120)
            # 合并运行结果及报错
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (120s)"
        except (FileNotFoundError, OSError) as e:
            return f"Error: {e}"

    # 检查操作的位置是否在工作目录内
    def safe_path(self, p: str) -> Path:
        path = (self.workdir / p).resolve()
        # 如果该目录在工作目录以外 报错
        if not path.is_relative_to(self.workdir):
            raise ValueError(f"Path escapes workspace: {p}")
        return path

    # 读文件
    def run_read(self, path: str, limit: int | None = None) -> str:
        try:
            lines = self.safe_path(path).read_text(encoding="utf-8").splitlines()
            if limit and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    # 写文件
    def run_write(self, path: str, content: str) -> str:
        try:
            file_path = self.safe_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    # 改文件
    def run_edit(self, path: str, old_text: str, new_text: str) -> str:
        try:
            file_path = self.safe_path(path)
            text = file_path.read_text(encoding="utf-8")
            if old_text not in text:
                return f"Error: text not found in {path}"
            file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
            return f"Edited {path}"
        except Exception as e:
            return f"Error: {e}"

    # 查文件
    def run_glob(self, pattern: str) -> str:
        import glob as g
        try:
            matches = sorted({      # 去重并排序
                match for match in g.glob(
                    pattern, root_dir=self.workdir, recursive=True)
                if (self.workdir / match).resolve().is_relative_to(self.workdir)
            })
            shown = matches[:200]
            if len(matches) > 200:
                shown.append("... (more matches omitted; narrow the pattern)")
            return "\n".join(shown) if shown else "(no matches)"
        except Exception as e:
            return f"Error: {e}"

    # 任务列表管理
    def run_todo_write(self, todos: list | str) -> str:
        try:
            output = self.todo.update(todos)
        except ValueError as e:
            return f"Error: {e}"
        print(f"\n\033[33m## Current Tasks\033[0m\n{output}")
        return output

    def execute_tool(self, block, handlers: dict) -> str:
        """Execute a tool between the PreToolUse and PostToolUse hooks."""
        blocked = self.hook_manager.trigger_hooks("PreToolUse", block)
        if blocked:
            return str(blocked)

        handler = handlers.get(block.name)
        try:
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
        except Exception as e:
            output = f"Error: {e}"

        self.hook_manager.trigger_hooks("PostToolUse", block, output)
        return str(output)
