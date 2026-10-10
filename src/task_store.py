from __future__ import annotations

import re
import json
import secrets

from dataclasses import dataclass, asdict
from pathlib import Path



# 匹配模式 task_加8位随机十六进制字符
TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")

@dataclass
class Task:
    id: str                 # task_加8位随机十六进制字符生成
    subject: str
    description: str
    status: str             # pending | in_progress | completed
    owner: str | None       # 负责当前任务的 Agent
    blockedBy: list[str]    # 依赖的任务 ID 列表


# 负责校验任务ID和读写JSON文件
class TaskStore:
    def __init__(self, workdir: Path, directory: Path):
        self.WORKDIR = Path(workdir).resolve()
        self.directory = Path(directory).resolve()
        self._root()

    # 拿到task_dir的绝对路径
    def _root(self, create: bool = False) -> Path:
        root = self.directory.resolve()
        if not root.is_relative_to(self.WORKDIR):
            raise ValueError("Task store escapes the workspace")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    # 把task_id拼进路径
    def _path(self, task_id: str, create_root: bool = False) -> Path:
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        root = self._root(create=create_root)
        path = (root / f"{task_id}.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        return path

    # 根据id判断存不存在同名任务
    def exists(self, task_id: str) -> bool:
        return self._path(task_id).is_file()

    # 创建任务并写入文件
    def create(self, subject: str, description: str = "") -> Task:
        subject = subject.strip()
        if not subject:
            raise ValueError("Task subject cannot be empty")

        self._root(create=True)
        for _ in range(100):
            task = Task(
                id=f"task_{secrets.token_hex(4)}",
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            try:
                with self._path(task.id, create_root=True).open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(asdict(task), handle, indent=2)
                return task
            except FileExistsError:
                continue
        raise RuntimeError("Could not allocate a unique task ID")

    # 检查两个任务是否直接或间接依赖
    def _depends_on(self, task_id: str, target_id: str) -> bool:
        pending = [task_id]
        visited = set()
        # 深度优先 沿着依赖关系查找
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self.load(current).blockedBy)
        return False

    # 根据任务id加载文件中的task数据
    def load(self, task_id: str) -> Task:
        data = json.loads(self._path(task_id).read_text(encoding="utf-8"))
        task = Task(**data)
        if task.id != task_id:
            raise ValueError(f"Task file ID does not match {task_id}")
        if task.status not in ("pending", "in_progress", "completed"):
            raise ValueError(f"Invalid task status: {task.status}")
        return task

    # 把task写入相应的file
    def save(self, task: Task) -> None:
        self._path(task.id, create_root=True).write_text(
            json.dumps(asdict(task), indent=2),
            encoding="utf-8",
        )

    # 添加依赖；全部检查通过后才保存，避免无效批次产生部分更新。
    def update_dependencies(self, task_id: str,
                            add_blocked_by: list[str]) -> Task:
        if not isinstance(add_blocked_by, list):
            raise ValueError("addBlockedBy must be a list of task IDs")
        task = self.load(task_id)
        if task.status != "pending" or task.owner is not None:
            raise ValueError(
                f"Task {task_id} dependencies can only be updated while "
                "pending and unowned"
            )

        dependencies = list(dict.fromkeys(add_blocked_by))
        for dependency in dependencies:
            if dependency == task_id:
                raise ValueError("Task cannot depend on itself")
            if not self.exists(dependency):
                raise ValueError(f"Dependency not found: {dependency}")
            if dependency not in task.blockedBy and self._depends_on(
                dependency, task_id
            ):
                raise ValueError(
                    f"Dependency cycle detected: {task_id} -> {dependency}"
                )

        task.blockedBy.extend(
            dependency for dependency in dependencies
            if dependency not in task.blockedBy
        )
        self.save(task)
        return task

    def list(self) -> list[Task]:
        if not self.directory.exists():
            return []
        root = self._root()
        return [self.load(path.stem) for path in sorted(root.glob("task_*.json"))]

    # 缺失或损坏的依赖也保持阻塞，不能被误当成已完成。
    def incomplete_dependencies(self, task: Task) -> list[str]:
        incomplete = []
        for dependency in task.blockedBy:
            try:
                if self.load(dependency).status != "completed":
                    incomplete.append(dependency)
            except (FileNotFoundError, ValueError):
                incomplete.append(dependency)
        return incomplete

    def can_start(self, task_id: str) -> bool:
        return not self.incomplete_dependencies(self.load(task_id))

    def claim(self, task_id: str, owner: str = "agent") -> Task:
        task = self.load(task_id)
        if task.status != "pending":
            raise ValueError(f"Task {task_id} is {task.status}, cannot claim")
        dependencies = self.incomplete_dependencies(task)
        if dependencies:
            raise ValueError(f"Blocked by: {dependencies}")
        task.owner = owner
        task.status = "in_progress"
        self.save(task)
        return task

    def complete(self, task_id: str, owner: str = "agent") -> tuple[Task, list[Task]]:
        task = self.load(task_id)
        if task.status != "in_progress":
            raise ValueError(f"Task {task_id} is {task.status}, cannot complete")
        if task.owner != owner:
            raise ValueError(f"Task {task_id} is owned by {task.owner}, not {owner}")

        ready_before = {
            candidate.id
            for candidate in self.list()
            if candidate.status == "pending"
            and candidate.blockedBy
            and self.can_start(candidate.id)
        }
        task.status = "completed"
        self.save(task)
        unblocked = [candidate for candidate in self.list()
                     if candidate.status == "pending"
                     and candidate.blockedBy
                     and candidate.id not in ready_before
                     and self.can_start(candidate.id)]
        return task, unblocked
