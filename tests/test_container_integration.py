"""Run the real image against synthetic services; no external LLM account required.

HERMES_TEST_IMAGE must name an already-built image. Each test owns its exact Docker
container/volume names and removes only those synthetic resources on completion.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import unittest
import uuid

from fixture_server import FixtureBackend, FixtureCase


@unittest.skipUnless(os.environ.get("HERMES_TEST_IMAGE"), "Set HERMES_TEST_IMAGE to a built runtime image")
class NativeContainerIntegration(unittest.TestCase):
    def setUp(self):
        self.image = os.environ["HERMES_TEST_IMAGE"]
        self.containers = []
        self.volumes = []
        self.fixture = FixtureBackend().__enter__()

    def tearDown(self):
        self.fixture.__exit__()
        for container in self.containers:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        for volume in self.volumes:
            subprocess.run(["docker", "volume", "rm", volume], check=True, capture_output=True)

    def volume(self):
        name = "penelopa-synthetic-" + uuid.uuid4().hex
        subprocess.run(["docker", "volume", "create", "--label", "io.penelopa.test=synthetic", name],
                       check=True, capture_output=True)
        self.volumes.append(name)
        return name

    def start(self, case, volume):
        self.fixture.cases.append(case)
        name = "penelopa-synthetic-" + uuid.uuid4().hex
        self.containers.append(name)
        command = ["docker", "run", "--name", name, "--add-host", "host.docker.internal:host-gateway",
                   "--user", "10001:10001", "--security-opt", "no-new-privileges:true",
                   "--cap-drop", "ALL", "--mount", f"type=volume,source={volume},target=/opt/data",
                   "--tmpfs", "/run/penelopa:uid=10001,gid=10001,mode=0700", "--pids-limit", "256"]
        for key, value in case.env(self.fixture.container_url).items():
            command.extend(["--env", f"{key}={value}"])
        command.append(self.image)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        return process

    def finish(self, case, process, *, pause_review=False):
        def wait_for(event, timeout):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if event.wait(0.2):
                    return True
                if process.poll() is not None:
                    return False
            return False

        if pause_review:
            if not wait_for(case.accepted, 90):
                process.kill()
                self.fail("No terminal MCP acceptance:\n" + process.communicate(timeout=10)[0])
            if not wait_for(case.review_started, 30):
                process.kill()
                self.fail("No native review after terminal ACK:\n" + process.communicate(timeout=10)[0])
            self.assertIsNone(process.poll(), "Container exited before native review completed")
        case.release_review.set()
        output, _ = process.communicate(timeout=150)
        self.assertEqual(process.returncode, 0, output)
        self.assertEqual(case.errors, [], output)
        self.assertEqual(case.terminal_calls, 1, output)
        self.assertTrue(case.review_started.is_set(), "No actual upstream background review:\n" + output)
        self.assertIn("list_user_sessions", case.mcp_calls)
        self.assertIn("read_session_events", case.mcp_calls)
        self.assertNotIn("report_failure", case.mcp_calls)
        return output

    def inspect_volume(self, volume):
        # Runs with network disabled. Includes SQLite bytes so secret-leak checks
        # cover native transcripts as well as markdown/config/log files.
        script = """
import json
from pathlib import Path
root = Path('/opt/data')
files = {str(p.relative_to(root)): p.read_bytes().decode('utf-8', errors='replace')
         for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
print(json.dumps(files))
"""
        result = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "10001:10001",
                                 "--mount", f"type=volume,source={volume},target=/opt/data,readonly",
                                 "--entrypoint", "/opt/hermes/.venv/bin/python", self.image, "-c", script],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def assert_persisted(self, case, files):
        all_text = json.dumps(files)
        self.assertIn(case.sentinel, files.get("hermes/memories/USER.md", ""))
        self.assertIn(case.review_sentinel, files.get("hermes/memories/MEMORY.md", ""))
        self.assertTrue(any(name.startswith("hermes/skills/fixture-" + case.label) for name in files))
        self.assertTrue(any(name.endswith("state.db") for name in files), "Missing genuine SessionDB")
        for secret in (case.task_token, case.lifecycle_token, case.provider_token):
            self.assertNotIn(secret, all_text)

    def test_real_harness_review_restart_and_user_isolation(self):
        alpha_volume, beta_volume = self.volume(), self.volume()
        alpha = FixtureCase(str(uuid.uuid4()), "alpha", force_fallback=True)
        self.finish(alpha, self.start(alpha, alpha_volume), pause_review=True)
        alpha_files = self.inspect_volume(alpha_volume)
        self.assert_persisted(alpha, alpha_files)

        alpha_next = FixtureCase(alpha.user_id, "alpha", prose_before_tools=True)
        self.finish(alpha_next, self.start(alpha_next, alpha_volume))
        prompts = json.dumps([r.get("messages") for r in alpha_next.provider_requests])
        self.assertIn(alpha.sentinel, prompts, "Native memory was not loaded on the next task")
        self.assertIn(alpha.review_sentinel, prompts, "Post-task learning did not survive restart")
        self.assertGreater(alpha_next.main_step, 6, "Prose alone incorrectly completed the task")

        beta = FixtureCase(str(uuid.uuid4()), "beta")
        self.finish(beta, self.start(beta, beta_volume))
        beta_files = self.inspect_volume(beta_volume)
        self.assert_persisted(beta, beta_files)
        self.assertNotIn(alpha.sentinel, json.dumps(beta_files))
        self.assertNotIn(alpha.sentinel, json.dumps(beta.provider_requests))
        self.assertNotIn(beta.sentinel, json.dumps(self.inspect_volume(alpha_volume)))

        schemas = [tool["function"]["name"] for request in alpha.provider_requests for tool in request.get("tools", [])]
        self.assertIn("memory", schemas)
        self.assertIn("skill_manage", schemas)
        self.assertIn("session_search", schemas)
        self.assertFalse({"terminal", "execute_code", "browser_navigate", "web_search"}.intersection(schemas))
        self.assertTrue(any(r.get("model") == "fixture-fallback" for r in alpha.provider_requests))
        self.assertTrue(alpha.heartbeats, "No lifecycle telemetry")

    def test_lost_terminal_ack_reconciles_without_republication(self):
        case = FixtureCase(str(uuid.uuid4()), "lost-ack", lost_ack=True)
        volume = self.volume()
        self.finish(case, self.start(case, volume), pause_review=True)
        self.assert_persisted(case, self.inspect_volume(volume))

    def test_idle_self_improvement_persists_without_recommendations(self):
        case = FixtureCase(str(uuid.uuid4()), "maintenance", maintenance=True)
        case.release_review.set()
        volume = self.volume()
        process = self.start(case, volume)
        output, _ = process.communicate(timeout=150)
        self.assertEqual(process.returncode, 0, output + "\nrequests=" + str([(len(json.dumps(r)), r.get("max_tokens")) for r in case.provider_requests]))
        self.assertEqual(case.terminal_calls, 1, output)
        self.assertNotIn("submit_recommendations", case.mcp_calls)
        self.assertIn("complete_self_improvement", case.mcp_calls)
        self.assertTrue(case.reports[-1]["complete"], output)
        self.assertEqual(case.reports[-1]["outcome"], "applied")
        files = self.inspect_volume(volume)
        self.assertIn(case.review_sentinel, files.get("hermes/memories/MEMORY.md", ""))
        next_case = FixtureCase(case.user_id, "maintenance-next")
        self.finish(next_case, self.start(next_case, volume))
        self.assertIn(case.review_sentinel, json.dumps(next_case.provider_requests))


if __name__ == "__main__":
    unittest.main()
