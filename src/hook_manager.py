import re
from pathlib import Path


class HookManager:
    """Manage agent hooks independently of the model client and tool handlers."""

    DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
    DESTRUCTIVE_COMMAND_WORD = re.compile(
        r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
    )
    DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.hooks = {
            "UserPromptSubmit": [],
            "PreToolUse": [],
            "PostToolUse": [],
            "Stop": [],
        }
        self.register_hook("UserPromptSubmit", self.context_inject_hook)
        self.register_hook("PreToolUse", self.permission_hook)
        self.register_hook("PreToolUse", self.log_hook)
        self.register_hook("PostToolUse", self.large_output_hook)
        self.register_hook("Stop", self.summary_hook)

    # 钩子注册表
    def register_hook(self, event: str, callback):
        self.hooks[event].append(callback)

    # 首个非 None 返回值会终止后续回调
    def trigger_hooks(self, event: str, *args):
        for callback in self.hooks[event]:
            result = callback(*args)
            if result is not None:
                return result
        return None

    def contains_destructive_command(self, command: str) -> bool:
        return bool(self.DESTRUCTIVE_COMMAND_WORD.search(command))

    def permission_hook(self, block):
        """PreToolUse: check commands and workspace access."""
        if block.name == "bash":
            command = block.input.get("command", "")
            for pattern in self.DENY_LIST:
                if pattern in command:
                    print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                    return "Permission denied by deny list"
            if self.contains_destructive_command(command) or any(
                kw in command for kw in self.DESTRUCTIVE
            ):
                print(f"\n\033[33m[permission] Potentially destructive command\033[0m")
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
        if block.name in ("read_file", "write_file", "edit_file"):
            path = block.input.get("path", "")
            if not (self.workdir / path).resolve().is_relative_to(self.workdir):
                print(f"\n\033[33m[permission] Access outside workspace\033[0m")
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
        return None

    def log_hook(self, block):
        """PreToolUse: log every tool call."""
        args_preview = str(list(block.input.values())[:2])[:60]
        print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
        return None

    def large_output_hook(self, block, output):
        """PostToolUse: warn on large output."""
        if len(str(output)) > 100000:
            print(f"\033[33m[HOOK] Large output from {block.name}: {len(str(output))} chars\033[0m")
        return None

    def context_inject_hook(self, query: str):
        print(f"\033[90m[HOOK] UserPromptSubmit: working in {self.workdir}\033[0m")
        return None

    def summary_hook(self, messages: list):
        tool_count = sum(
            1 for m in messages
            for b in (m.get("content") if isinstance(m.get("content"), list) else [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        )
        print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
        return None
