import json
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_store import TASK_ID_PATTERN, TaskStore


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / ".tasks"
        self.store = TaskStore(self.root, self.directory)

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.directory.glob("*.json")}

    def path(self, task):
        return self.directory / f"{task.id}.json"

    def test_empty_store_does_not_create_directory(self):
        self.assertEqual(self.store.list(), [])
        self.assertFalse(self.store.exists("task_12345678"))
        self.assertFalse(self.directory.exists())

    def test_chinese_records_and_restart(self):
        task = self.store.create("  编写接口  ", "实现任务管理接口")
        self.assertRegex(task.id, TASK_ID_PATTERN)
        self.assertEqual(task.subject, "编写接口")
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.owner)
        self.assertEqual(task.blockedBy, [])
        self.assertEqual(json.loads(self.path(task).read_text(encoding="utf-8")), asdict(task))
        restarted = TaskStore(self.root, self.directory)
        self.assertEqual(restarted.load(task.id), task)
        self.store.claim(task.id)
        self.assertEqual(restarted.load(task.id).owner, "agent")
        restarted.complete(task.id)
        self.assertEqual(self.store.load(task.id).status, "completed")

    def test_workspace_boundary_checked_before_creating_directory(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        outside = self.root / "outside"
        with self.assertRaisesRegex(ValueError, "escapes the workspace"):
            TaskStore(workspace, outside)
        self.assertFalse(outside.exists())
        store = TaskStore(workspace / ".." / "workspace", workspace / ".tasks")
        self.assertEqual(store.WORKDIR, workspace.resolve())
        store.directory = outside
        with self.assertRaises(ValueError):
            store.create("Rejected")
        self.assertFalse(outside.exists())

    def test_invalid_ids_cannot_escape_store(self):
        for task_id in ("../outside", "task_12345678/../other", "task_123", "task_ABCDEFGH", "", None, []):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                self.store.load(task_id)
        self.assertFalse(self.directory.exists())

    def test_invalid_create_preserves_store(self):
        for subject, description in (("", ""), (" ", "")):
            with self.subTest(subject=subject), self.assertRaises(ValueError):
                self.store.create(subject, description)
        self.assertFalse(self.directory.exists())

    def test_dependencies_deduplicate_and_list_is_sorted(self):
        first = self.store.create("First")
        second = self.store.create("Second")
        task = self.store.create("Dependent")
        updated = self.store.update_dependencies(task.id, [second.id, first.id, second.id])
        self.assertEqual(updated.blockedBy, [second.id, first.id])
        self.assertEqual(self.store.update_dependencies(task.id, [first.id]).blockedBy, updated.blockedBy)
        self.assertEqual([item.id for item in self.store.list()], sorted([first.id, second.id, task.id]))

    def test_invalid_dependency_batches_do_not_partially_update(self):
        first = self.store.create("First")
        task = self.store.create("Dependent")
        before = self.snapshot()
        batches = ([first.id, "task_00000000"], [first.id, task.id],
                   [first.id, "../outside"], [first.id, []], "not a list")
        for batch in batches:
            with self.subTest(batch=batch), self.assertRaises((ValueError, TypeError)):
                self.store.update_dependencies(task.id, batch)
            self.assertEqual(self.snapshot(), before)

    def test_direct_and_transitive_cycles_are_rejected(self):
        first = self.store.create("First")
        second = self.store.create("Second")
        third = self.store.create("Third")
        self.store.update_dependencies(second.id, [first.id])
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.store.update_dependencies(first.id, [second.id])
        self.assertEqual(self.snapshot(), before)
        self.store.update_dependencies(third.id, [second.id])
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.store.update_dependencies(first.id, [third.id])
        self.assertEqual(self.snapshot(), before)

    def test_dependency_chain_and_newly_unblocked_tasks(self):
        schema = self.store.create("Schema")
        api = self.store.create("API")
        tests = self.store.create("Tests")
        self.store.update_dependencies(api.id, [schema.id])
        self.store.update_dependencies(tests.id, [api.id])
        self.assertTrue(self.store.can_start(schema.id))
        self.assertFalse(self.store.can_start(api.id))
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Blocked by"):
            self.store.claim(api.id)
        self.assertEqual(self.snapshot(), before)
        self.store.claim(schema.id)
        completed, unblocked = self.store.complete(schema.id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual([task.id for task in unblocked], [api.id])
        self.assertFalse(self.store.can_start(tests.id))
        self.store.claim(api.id)
        _, unblocked = self.store.complete(api.id)
        self.assertEqual([task.id for task in unblocked], [tests.id])
        self.store.claim(tests.id)
        self.assertEqual(self.store.complete(tests.id)[1], [])

    def test_multiple_dependencies_and_no_repeated_unlock_notice(self):
        first = self.store.create("First")
        second = self.store.create("Second")
        dependent = self.store.create("Dependent")
        already_ready = self.store.create("Already ready")
        self.store.update_dependencies(dependent.id, [first.id, second.id])
        self.store.update_dependencies(already_ready.id, [first.id])
        self.store.claim(first.id)
        self.store.claim(second.id)
        self.assertEqual(sum(task.status == "in_progress" for task in self.store.list()), 2)
        _, unblocked = self.store.complete(first.id)
        self.assertEqual([task.id for task in unblocked], [already_ready.id])
        self.assertEqual(self.store.incomplete_dependencies(self.store.load(dependent.id)), [second.id])
        _, unblocked = self.store.complete(second.id)
        self.assertEqual([task.id for task in unblocked], [dependent.id])

    def test_invalid_transitions_and_owner_preserve_records(self):
        task = self.store.create("Work")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "cannot complete"):
            self.store.complete(task.id)
        self.assertEqual(self.snapshot(), before)
        self.store.claim(task.id, owner="other")
        before = self.snapshot()
        for operation in (lambda: self.store.claim(task.id),
                          lambda: self.store.complete(task.id),
                          lambda: self.store.update_dependencies(task.id, [])):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(self.snapshot(), before)
        self.store.complete(task.id, owner="other")
        before = self.snapshot()
        for operation in (lambda: self.store.complete(task.id, owner="other"),
                          lambda: self.store.claim(task.id),
                          lambda: self.store.update_dependencies(task.id, [])):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(self.snapshot(), before)

    def test_pending_owned_task_dependencies_cannot_be_updated(self):
        task = self.store.create("Owned")
        task.owner = "other"
        self.store.save(task)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.store.update_dependencies(task.id, [])
        self.assertEqual(self.snapshot(), before)

    def test_missing_or_corrupt_dependencies_remain_blocked(self):
        dependency = self.store.create("Dependency")
        task = self.store.create("Dependent")
        self.store.update_dependencies(task.id, [dependency.id])
        dependent_before = self.path(task).read_bytes()
        corrupt_records = ["{", json.dumps({**asdict(dependency), "status": "invalid"}),
                           json.dumps({**asdict(dependency), "id": "task_00000000"}),
                           bytes([255, 254])]
        self.path(dependency).unlink()
        for content in [None, *corrupt_records]:
            with self.subTest(content=content):
                if isinstance(content, bytes):
                    self.path(dependency).write_bytes(content)
                elif content is not None:
                    self.path(dependency).write_text(content, encoding="utf-8")
                self.assertFalse(self.store.can_start(task.id))
                self.assertEqual(self.store.incomplete_dependencies(self.store.load(task.id)), [dependency.id])
                with self.assertRaisesRegex(ValueError, "Blocked by"):
                    self.store.claim(task.id)
                self.assertEqual(self.path(task).read_bytes(), dependent_before)

    def test_corrupt_records_report_errors_before_completion_is_saved(self):
        task = self.store.create("Work")
        self.store.claim(task.id)
        corrupt = self.store.create("Corrupt")
        self.path(corrupt).write_text("{}", encoding="utf-8")
        before = self.snapshot()
        with self.assertRaises(TypeError):
            self.store.list()
        with self.assertRaises(TypeError):
            self.store.complete(task.id)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
