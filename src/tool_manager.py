import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from background_manager import (
    _format_bash_result,
    _run_bash_process,
    should_run_background,
    start_background_task,
)
from hook_manager import HookManager
from skill_loader import SkillLoader
from task_store import TASK_ID_PATTERN, TaskStore


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

# 主 agent 的持久化任务工具；task 工具仍用于子 agent 委派。
TASK_TOOLS = [
    {"name": "create_task", "description": "Create a task and return its runtime-generated ID.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"], "additionalProperties": False}},
    {"name": "update_task", "description": "Add dependencies using IDs returned by create_task.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string", "pattern": TASK_ID_PATTERN.pattern}, "addBlockedBy": {"type": "array", "items": {"type": "string", "pattern": TASK_ID_PATTERN.pattern}, "minItems": 1}}, "required": ["task_id", "addBlockedBy"], "additionalProperties": False}},
    {"name": "list_tasks", "description": "List tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_task", "description": "Get a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a pending task whose dependencies are complete.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete the task claimed by this agent.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
]

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
                 skill_loader: SkillLoader, task_handler: Callable[[str], str],
                 task_store: TaskStore):
        self.workdir = workdir
        self.hook_manager = hook_manager
        self.task_store = task_store

        # 子 agent 仅使用基础能力，复制容器以便独立组装。
        self.sub_tools = list(BASE_TOOLS)
        self.sub_handlers = {
            "bash": self.run_bash,
            "read_file": self.run_read,
            "write_file": self.run_write,
            "edit_file": self.run_edit,
            "glob": self.run_glob,
        }
        # 仅主 agent 暴露后台参数，不修改子 agent 使用的基础定义。
        main_base_tools = deepcopy(BASE_TOOLS)
        main_base_tools[0]["input_schema"]["properties"]["run_in_background"] = {
            "type": "boolean"
        }
        self.main_tools = [*main_base_tools, SKILL_TOOL, *TASK_TOOLS, TASK_TOOL, COMPACT_TOOL]
        # compact 由主循环在工具批次结束后处理，不进入普通分发器。
        self.main_handlers = {
            **self.sub_handlers,
            "load_skill": skill_loader.load,
            "create_task": self.run_create_task,
            "update_task": self.run_update_task,
            "list_tasks": self.run_list_tasks,
            "get_task": self.run_get_task,
            "claim_task": self.run_claim_task,
            "complete_task": self.run_complete_task,
            "task": task_handler,
        }

    # 执行command并返回结果给调用方
    def run_bash(self, command: str, run_in_background: bool = False) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        # 危险操作检查
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        return _format_bash_result(*_run_bash_process(command))

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

    # 任务状态规则交给 TaskStore，这里只处理工具参数及返回文本。
    def run_create_task(self, subject: str, description: str = "") -> str:
        task = self.task_store.create(subject, description)
        print(f"  [create] {task.subject}")
        return f"Created {task.id}: {task.subject}"

    def run_update_task(self, task_id: str, addBlockedBy: list[str]) -> str:
        task = self.task_store.update_dependencies(task_id, addBlockedBy)
        dependencies = ", ".join(task.blockedBy) or "(none)"
        print(f"  [update] {task.subject} blockedBy: {dependencies}")
        return f"Updated {task.id} blockedBy: {dependencies}"

    def run_list_tasks(self) -> str:
        tasks = self.task_store.list()
        if not tasks:
            return "No tasks. Use create_task to add some."
        lines = []
        for task in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task.status, "[?]")
            dependencies = f" (blockedBy: {', '.join(task.blockedBy)})" if task.blockedBy else ""
            owner = f" [{task.owner}]" if task.owner else ""
            lines.append(f"{marker} {task.id}: {task.subject} [{task.status}]{owner}{dependencies}")
        return "\n".join(lines)

    def run_get_task(self, task_id: str) -> str:
        return json.dumps(asdict(self.task_store.load(task_id)), indent=2)

    def run_claim_task(self, task_id: str) -> str:
        task = self.task_store.claim(task_id, owner="agent")
        print(f"  [claim] {task.subject} -> in_progress (owner: agent)")
        return f"Claimed {task.id} ({task.subject})"

    def run_complete_task(self, task_id: str) -> str:
        task, unblocked = self.task_store.complete(task_id, owner="agent")
        print(f"  [complete] {task.subject}")
        message = f"Completed {task.id} ({task.subject})"
        if unblocked:
            message += f"\nUnblocked: {', '.join(candidate.subject for candidate in unblocked)}"
            print(f"  [unblocked] {', '.join(candidate.subject for candidate in unblocked)}")
        return message

    def execute_tool(self, block, handlers: dict) -> str:
        """Execute a tool between the PreToolUse and PostToolUse hooks."""
        blocked = self.hook_manager.trigger_hooks("PreToolUse", block)
        if blocked is not None:
            return str(blocked)

        try:
            if handlers is self.main_handlers and should_run_background(block.name, block.input):
                # 后台分支也保留 run_bash 原有的危险命令检查。
                dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
                command = block.input.get("command")
                if isinstance(command, str) and any(d in command for d in dangerous):
                    output = "Error: Dangerous command blocked"
                else:
                    task_id = start_background_task(block)
                    output = (
                        f"[Background task {task_id} started] "
                        "The result will be collected on a later turn."
                    )
            else:
                handler = handlers.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
        except Exception as e:
            output = f"Error: {e}"

        self.hook_manager.trigger_hooks("PostToolUse", block, output)
        return str(output)
