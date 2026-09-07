"""Revision-pinned checks executed inside the actual shipped image."""

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from penelopa_runtime.native import ALLOWED_NAMES, install_policy, permitted


@unittest.skipUnless(importlib.util.find_spec("run_agent"), "Requires the pinned upstream image")
class NativeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_policy()

    def test_native_signatures_and_dispatch(self):
        from agent.background_review import (
            prepare_background_review_run,
            spawn_background_review_thread,
        )
        from hermes_cli.goals import GoalManager
        from run_agent import AIAgent
        from tools.registry import registry

        self.assertIn("skip_background_review", inspect.signature(AIAgent).parameters)
        self.assertIn("review_run", inspect.signature(spawn_background_review_thread).parameters)
        self.assertTrue(callable(prepare_background_review_run))
        self.assertTrue(callable(GoalManager.evaluate_after_turn))
        self.assertFalse(permitted("terminal"))
        self.assertFalse(permitted("session_search", {"profile": "other-user"}))
        self.assertFalse(permitted("session_search", {"profile": "default"}))
        self.assertFalse(permitted("session_search", {"session_id": "foreign-user/id"}))
        called = []
        registry.register("terminal", "terminal", {}, lambda args: called.append(True))
        result = registry.dispatch("terminal", {"command": "echo forbidden"})
        self.assertIn("denied", result.lower())
        self.assertEqual(called, [])
        self.assertTrue(set(registry.get_all_tool_names()) <= ALLOWED_NAMES)

    def test_session_miss_never_scans_profiles(self):
        from tools import session_search_tool

        with patch.object(session_search_tool, "_read_session", return_value='{"success":false}'):
            result = session_search_tool.session_search(session_id="missing-id", db=object())
        self.assertFalse(json.loads(result)["success"])
        with self.assertRaises(ValueError):
            session_search_tool._resolve_profile_db("default")

    def test_native_memory_skill_schemas(self):
        from model_tools import get_tool_definitions

        definitions = get_tool_definitions(
            enabled_toolsets=["memory", "skills", "session_search"], quiet_mode=True
        )
        schemas = {item["function"]["name"]: item["function"]["parameters"] for item in definitions}
        self.assertIn("target", schemas["memory"]["required"])
        self.assertIn("operations", schemas["skill_manage"]["required"])
        self.assertNotIn("terminal", json.dumps(schemas))

    def test_native_resume_follows_committed_compression_child(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as root:
            db = SessionDB(Path(root) / "state.db")
            try:
                db.create_session("parent", "cron")
                db.append_message("parent", "user", "old context")
                db.end_session("parent", "compression")
                db.create_session("child", "cron", parent_session_id="parent")
                db.append_message("child", "assistant", "latest committed context")
                tip = db.resolve_resume_session_id("parent")
                self.assertEqual(tip, "child")
                self.assertEqual(db.get_messages(tip)[0]["content"], "latest committed context")
                db.create_session(
                    "delegate",
                    "tool",
                    parent_session_id="child",
                    model_config={"_delegate_from": "child"},
                )
                db.append_message("delegate", "assistant", "unrelated delegated context")
                self.assertEqual(db.resolve_resume_session_id("parent"), "child")
            finally:
                db.close()

    def test_native_skill_cannot_execute_shell_or_rewrite_config(self):
        # Fresh native imports avoid cached profile paths from other contracts.
        script = r"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from penelopa_runtime.config import write_managed_config
from penelopa_runtime.native import install_policy
home = Path(os.environ["HERMES_HOME"])
settings = SimpleNamespace(home=home, model="fixture-primary", internal_timeout=30)
write_managed_config(settings, "http://127.0.0.1:12345")
before = (home / "config.yaml").read_text()
install_policy()
from model_tools import get_tool_definitions
from tools.registry import registry
get_tool_definitions(enabled_toolsets=["memory", "skills", "session_search"], quiet_mode=True)
content = "---\nname: audit-skill\ndescription: Synthetic audit notes\n---\nLiteral !`echo NEVER-EXECUTE`"
created = registry.dispatch("skill_manage", {
    "name": "audit-skill", "operations": [{"action": "create", "content": content}],
})
assert json.loads(created)["success"], created
with patch("agent.skill_preprocessing.run_inline_shell", side_effect=AssertionError("shell bypass")) as shell:
    viewed = registry.dispatch("skill_view", {"name": "audit-skill"})
    assert json.loads(viewed)["success"], viewed
    assert shell.call_count == 0
escaped = registry.dispatch("skill_manage", {
    "name": "audit-skill", "operations": [{"action": "write_file",
        "file_path": "../../config.yaml", "file_content": "inline_shell: true"}],
})
assert not json.loads(escaped)["success"], escaped
assert (home / "config.yaml").read_text() == before
"""
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run(
                [sys.executable, "-c", script],
                env={**os.environ, "HERMES_HOME": root, "HERMES_WRITE_SAFE_ROOT": root},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
