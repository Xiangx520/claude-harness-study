import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hook_manager import HookManager
from memory_store import MemoryStore, RECALL_CHAR_LIMIT


def response(value, stop_reason="end_turn"):
    text = value if isinstance(value, str) else json.dumps(value)
    return SimpleNamespace(content=[{"type": "text", "text": text}], stop_reason=stop_reason)


def record(name="Python preference", **changes):
    value = {"name": name, "type": "user", "scope": "persistent",
             "description": "Preferred Python language", "body": "Use Python for examples."}
    value.update(changes)
    return value


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.directory = self.root / ".memory"
        self.index = self.directory / "MEMORY.md"
        self.create = Mock()
        self.client = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.store = MemoryStore(self.root, self.directory, self.index, self.client, "test-model")
        self.messages = [{"role": "user", "content": "Please remember my Python preference"}]

    def seed(self, count=1):
        for index in range(count):
            self.store.write_memory_file(f"Record {index}", "project", f"Description {index}", f"Body {index}")

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.directory.glob("*.md")}

    def quiet(self, callback, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return callback(*args)

    def test_storage_index_and_restart(self):
        path = self.store.write_memory_file("Python preference", "user", "Python: preferred", "Use Python.")
        self.assertEqual(path.name, "python-preference.md")
        self.assertIn("[Python preference](python-preference.md)", self.store.read_memory_index())
        restarted = MemoryStore(self.root, self.directory, self.index)
        self.assertEqual(restarted.list_memory_files()[0]["body"], "Use Python.")
        metadata, body = restarted.parse_frontmatter(restarted.read_memory_file(path.name))
        self.assertEqual(metadata["description"], "Python: preferred")
        self.assertEqual(body.strip(), "Use Python.")

    def test_frontmatter_and_json_parsing(self):
        metadata, body = MemoryStore.parse_frontmatter('---\r\nname: "a---b"\r\ntype: user\r\n---\r\nText')
        self.assertEqual(metadata["name"], "a---b")
        self.assertEqual(body, "Text")
        self.assertEqual(MemoryStore.parse_frontmatter("---\nname: [\n---\nText")[0], {})
        self.assertEqual(MemoryStore.extract_json_array("```json\n[0, 1]\n```"), [0, 1])
        self.assertEqual(MemoryStore.extract_json_array('[{"body": [1, 2]}'), [])

    def test_workspace_and_index_validation(self):
        for filename in ("../other.md", "nested/file.md", "nested\\file.md", "MEMORY.md", "memory.md", "", ".", ".."):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.store.memory_path(filename)
        with self.assertRaises(ValueError):
            MemoryStore(self.root, self.root.parent / "outside", self.index)
        with self.assertRaises(ValueError):
            MemoryStore(self.root, self.directory, self.root / "MEMORY.md")
        with self.assertRaises(ValueError):
            self.store.write_memory_file("MEMORY", "user", "Description", "Body")
        self.assertFalse(self.directory.exists())

    def test_missing_and_corrupt_records(self):
        self.assertEqual(self.store.read_memory_index(), "")
        self.assertIsNone(self.store.read_memory_file("missing.md"))
        self.assertEqual(self.store.list_memory_files(), [])
        self.seed()
        (self.directory / "broken.md").write_bytes(bytes([255, 254, 0]))
        self.assertIsNone(self.store.read_memory_file("broken.md"))
        self.assertEqual(len(self.store.list_memory_files()), 1)
        self.store.rebuild_memory_index()
        self.assertNotIn("broken.md", self.store.read_memory_index())

    def test_recall_indices_deduplication_and_budget(self):
        self.seed(3)
        self.create.return_value = response([True, -1, 99, "0", 2, 2, 0, 1])
        self.assertEqual(self.store.select_relevant_memories(self.messages, 2), ["record-2.md", "record-0.md"])
        self.assertEqual(self.store.select_relevant_memories(self.messages, 0), [])
        self.store.write_memory_file("Long", "project", "Large record", "x" * 30000)
        self.create.return_value = response([0, 1])
        recalled = json.loads(self.store.load_memories(self.messages))
        self.assertLessEqual(sum(len(item["content"]) for item in recalled), RECALL_CHAR_LIMIT)
        self.assertEqual(recalled[0]["source"], "long.md")

    def test_recall_fallback_and_no_model(self):
        self.store.write_memory_file("Python preference", "user", "Python language", "Use Python.")
        self.create.side_effect = RuntimeError("offline")
        self.assertEqual(self.store.select_relevant_memories(self.messages), ["python-preference.md"])
        local = MemoryStore(self.root, self.directory, self.index)
        self.assertEqual(local.select_relevant_memories(self.messages), ["python-preference.md"])
        self.assertEqual(local.extract_memories(self.messages), 0)
        self.assertEqual(local.consolidate_memories(), 0)

    def test_extraction_filters_and_duplicate_records(self):
        self.create.return_value = response([
            record(), record(), record("Temporary", scope="current_task"),
            record("One off", body="Use it just this time"),
            record("Invalid", type="unknown"), record("Null", body=None),
        ])
        self.assertEqual(self.quiet(self.store.extract_memories, self.messages), 1)
        self.assertEqual(len(self.store.list_memory_files()), 1)
        self.assertEqual(self.quiet(self.store.extract_memories, self.messages), 0)
        self.assertTrue(self.index.exists())

    def test_extraction_model_failure(self):
        self.create.side_effect = RuntimeError("offline")
        self.assertEqual(self.quiet(self.store.extract_memories, self.messages), 0)
        self.assertFalse(self.directory.exists())

    def test_system_preserves_base_prompt(self):
        self.seed()
        system = self.store.augment_system("Skills and task instructions", "recalled data")
        self.assertTrue(system.startswith("Skills and task instructions"))
        self.assertIn("current user request takes priority", system)
        self.assertIn("recalled data", system)
        self.assertIn("record-0.md", system)

    def test_text_helpers_ignore_tool_results(self):
        messages = [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": [SimpleNamespace(type="text", text="answer")]},
            {"role": "user", "content": [{"type": "tool_result", "content": "secret output"}]},
        ]
        self.assertEqual(self.store.recent_user_text(messages), "request")
        self.assertEqual(self.store.dialogue_text(messages), "user: request\nassistant: answer")
        self.assertEqual(self.store.recent_user_text(messages, 0), "")

    def test_consolidation_threshold_and_success(self):
        self.seed(9)
        self.assertEqual(self.store.consolidate_memories(), 0)
        self.create.assert_not_called()
        self.seed(10)
        self.create.return_value = response([record("Merged", description="Merged facts", body="Stable project facts.")])
        self.assertEqual(self.quiet(self.store.consolidate_memories), 1)
        self.assertEqual([r["filename"] for r in self.store.list_memory_files()], ["merged.md"])
        self.assertIn("merged.md", self.store.read_memory_index())

    def test_invalid_consolidation_preserves_files(self):
        self.seed(10)
        before = self.snapshot()
        cases = [response([]), response([record(), {"name": "bad"}]),
                 response([record(), record()]), response([record("MEMORY")]),
                 response([record()], stop_reason="max_tokens"), response("invalid json"),
                 response([record(f"Name {i}") for i in range(31)])]
        for result in cases:
            with self.subTest(result=result):
                self.create.return_value = result
                self.assertEqual(self.quiet(self.store.consolidate_memories), 0)
                self.assertEqual(self.snapshot(), before)

    def test_consolidation_write_failure_rolls_back(self):
        self.seed(10)
        before = self.snapshot()
        self.create.return_value = response([record("Merged")])
        original = Path.write_text
        def fail_merged(path, *args, **kwargs):
            if path.name == "merged.md":
                raise OSError("simulated disk error")
            return original(path, *args, **kwargs)
        with patch.object(Path, "write_text", fail_merged):
            self.assertEqual(self.quiet(self.store.consolidate_memories), 0)
        self.assertEqual(self.snapshot(), before)

    def test_consolidation_preserves_unreadable_files(self):
        self.seed(10)
        broken = self.directory / "broken.md"
        broken.write_bytes(bytes([255, 254, 0]))
        self.create.return_value = response([record("Merged")])
        self.assertEqual(self.quiet(self.store.consolidate_memories), 1)
        self.assertEqual(broken.read_bytes(), bytes([255, 254, 0]))

    def test_parent_agent_memory_lifecycle(self):
        import anthropic
        with patch.object(anthropic, "Anthropic", return_value=self.client), patch("dotenv.load_dotenv"), patch.dict(os.environ, {"MODEL_ID": "test-model"}):
            import loop_agent
        self.store.write_memory_file("Python preference", "user", "Python language", "Use Python.")
        answer = SimpleNamespace(content=[SimpleNamespace(type="text", text="Done")])
        self.create.side_effect = [response([0]), answer, response([record("Formatting", description="Formatting style", body="Prefer readable formatting.")])]
        compactor = SimpleNamespace(prepare=lambda messages, request: messages)
        with patch.object(loop_agent, "client", self.client), patch.object(loop_agent, "MEMORY_STORE", self.store), patch.object(loop_agent, "COMPACTOR", compactor), patch.object(loop_agent, "HOOK_MANAGER", HookManager(self.root)):
            self.quiet(loop_agent.agent_loop, self.messages, "request")
        self.assertEqual(self.create.call_count, 3)
        system = self.create.call_args_list[1].kwargs["system"]
        self.assertIn("Use Python.", system)
        self.assertIn("Skills available", system)
        self.assertEqual(len(self.store.list_memory_files()), 2)
        self.assertEqual(self.messages[-1]["content"][0].text, "Done")


if __name__ == "__main__":
    unittest.main()
