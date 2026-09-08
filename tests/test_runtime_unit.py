"""Transport/lifecycle unit tests; no upstream installation or real credentials."""

import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from penelopa_runtime.broker import Broker, CapabilityRevoked, TransportError, exchange
from penelopa_runtime.config import Settings, bootstrap, endpoint
from penelopa_runtime.state import RuntimeState, atomic_json


def settings(root):
    home = Path(root) / "hermes"
    home.mkdir()
    return Settings(
        task_id="00000000-0000-0000-0000-000000000001",
        user_id="00000000-0000-0000-0000-000000000002",
        claim_version=1,
        api_base="http://backend.invalid",
        provider_base="http://provider.invalid/v1",
        model="primary",
        fallback_models=("fallback",),
        task_token="test-task-secret",
        lifecycle_token="test-lifecycle-secret",
        provider_token="test-provider-secret",
        home=home,
    )


class BootstrapUnit(unittest.TestCase):
    def test_nonroot_bootstrap_creates_and_preserves_dedicated_workspace(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ),
            patch("penelopa_runtime.config.os.geteuid", return_value=10001),
        ):
            home = Path(root) / "hermes"
            previous_cwd = Path.cwd()
            bootstrap(home)
            workspace = Path(root) / "workspace"
            self.assertTrue(workspace.is_dir())
            self.assertNotEqual(workspace, home)
            retained = workspace / "retained.txt"
            retained.write_text("keep existing user data")
            bootstrap(home)
            self.assertEqual(retained.read_text(), "keep existing user data")
            self.assertEqual(Path.cwd(), previous_cwd)

    def test_nonroot_bootstrap_rejects_workspace_symlink(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ),
            patch("penelopa_runtime.config.os.geteuid", return_value=10001),
        ):
            outside = Path(root) / "outside"
            outside.mkdir()
            (Path(root) / "workspace").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "workspace.*symlink"):
                bootstrap(Path(root) / "hermes")
            self.assertEqual(list(outside.iterdir()), [])


class RuntimeUnit(unittest.TestCase):
    def test_maintenance_budget_survives_restart_and_bounds_each_call(self):
        brief = {"task_kind": "self_improvement", "deadline": (datetime.now(timezone.utc) + timedelta(seconds=180)).isoformat(),
            "budget": {"calls": 2, "input_tokens": 100000, "output_tokens": 8000}}
        self.broker.configure_task(brief)
        payload = {"model": "primary", "messages": [{"role": "user", "content": "review"}], "max_tokens": 32000}
        self.assertEqual(self.broker.reserve_call(payload)["max_tokens"], 2048)
        self.broker.reserve_call(payload)
        restarted = Broker(self.settings, self.state)
        self.addCleanup(restarted.server.server_close)
        restarted.configure_task(brief)
        with self.assertRaises(TransportError):
            restarted.reserve_call(payload)
        self.assertEqual(self.state.error_code, "budget_exhausted")
        self.assertNotIn("submit_recommendations", restarted.allowed_tools())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = settings(self.temp.name)
        self.state = RuntimeState(self.settings)
        self.broker = Broker(self.settings, self.state)
        self.addCleanup(self.broker.server.server_close)

    def test_aggregate_does_not_drop_calls_after_128(self):
        for _ in range(150):
            self.state.record(
                "analysis",
                "fallback",
                1000,
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )
        result = self.state.snapshot()
        self.assertEqual(result["llm_requests"], 150)
        self.assertEqual(len(result["generation_stages"]), 1)
        self.assertEqual(result["generation_stages"][0]["usage"]["total_tokens"], 1800)
        self.assertEqual(result["generation_stages"][0]["duration_seconds"], 150000)
        self.assertEqual(self.state.terminal_meta()["io.auto-improve/hermes"]["model"], "fallback")
        restored = RuntimeState(self.settings)
        self.assertEqual(restored.snapshot()["generation_stages"], result["generation_stages"])
        self.assertEqual(restored.requests, 150)
        self.assertEqual(restored.last_model, "fallback")
        self.assertIsNone(restored.accepted)
        for secret in (
            self.settings.provider_token,
            self.settings.task_token,
            self.settings.lifecycle_token,
        ):
            self.assertNotIn(secret, self.state.path.read_text())

    def test_internal_model_tools_and_forged_metadata_are_denied(self):
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "heartbeat_task", "arguments": {}},
        }
        with patch.object(self.broker, "remote_rpc") as remote:
            self.assertIn("error", self.broker.mcp(request))
            remote.assert_not_called()
        request["params"] = {
            "name": "read_session_events",
            "arguments": {},
            "_meta": {"forged": True},
        }
        with patch.object(self.broker, "remote_rpc", return_value={"result": {}}) as remote:
            self.broker.mcp(request)
            self.assertNotIn("_meta", remote.call_args.args[0]["params"])

    def test_receipt_is_bound_to_exact_claim(self):
        with patch.object(
            self.broker,
            "lifecycle",
            return_value={
                "task_id": self.settings.task_id,
                "claim_version": 2,
                "accepted": True,
                "status": "SUCCEEDED",
                "receipt": {},
            },
        ):
            with self.assertRaises(CapabilityRevoked):
                self.broker.receipt()
        self.assertTrue(self.state.stop.is_set())
        self.assertIsNone(self.state.accepted)

    def test_server_lease_expiry_stops_independently_of_http_calls(self):
        self.broker.update_task_lease(
            {
                "result": {
                    "structuredContent": {
                        "task_id": self.settings.task_id,
                        "status": "RUNNING",
                        "lease_expires_at": (
                            datetime.now(timezone.utc) + timedelta(seconds=0.2)
                        ).isoformat(),
                    }
                }
            }
        )
        self.assertTrue(self.broker.lease_ready.is_set())
        self.broker.lease_thread.start()
        try:
            self.assertTrue(self.state.stop.wait(1))
            self.assertEqual(self.state.error_code, "task_lease_expired")
        finally:
            self.broker.done.set()
            self.broker.lease_thread.join()

    def test_missing_lease_does_not_unlock_model_start(self):
        with self.assertRaises(TransportError):
            self.broker.update_task_lease(
                {
                    "result": {
                        "structuredContent": {
                            "task_id": self.settings.task_id,
                            "status": "RUNNING",
                        }
                    }
                }
            )
        self.assertFalse(self.broker.lease_ready.is_set())

    def test_accepted_receipt_closes_remote_tools(self):
        self.state.accepted = {"task_id": self.settings.task_id, "status": "SUCCEEDED"}
        with patch.object(self.broker, "remote_rpc") as remote:
            result = self.broker.mcp(
                {
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "submit_recommendations", "arguments": {"forged": True}},
                }
            )
            self.assertFalse(result["result"]["isError"])
            self.assertIn(
                "error",
                self.broker.mcp(
                    {
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "read_session_events", "arguments": {}},
                    }
                ),
            )
            remote.assert_not_called()

    def test_fallback_and_exhaustion_are_physical_attempts(self):
        usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
        with patch(
            "penelopa_runtime.broker.exchange",
            side_effect=[
                TransportError("http_524"),
                (200, {"Content-Type": "application/json"}, json.dumps({"usage": usage}).encode()),
            ],
        ) as call:
            self.broker.provider("/provider/goal_check/v1/chat/completions", {"model": "primary"})
            self.assertEqual(call.call_args.args[2]["model"], "fallback")
            self.assertEqual(call.call_args.kwargs["headers"]["X-Penelopa-Stage"], "goal_check")
        self.assertEqual(self.state.requests, 2)
        with patch("penelopa_runtime.broker.exchange", side_effect=TransportError("http_503")):
            with self.assertRaises(TransportError):
                self.broker.provider("/provider/analysis/v1/chat/completions", {"model": "primary"})
        self.assertTrue(self.state.stop.is_set())
        self.assertEqual(self.state.requests, 4)

    def test_managed_checkpoint_rejects_symlink(self):
        outside = Path(self.temp.name) / "outside"
        outside.write_text("untouched")
        link = Path(self.temp.name) / "runtime.json"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            atomic_json(link, {"bad": True})
        self.assertEqual(outside.read_text(), "untouched")

    def test_endpoint_rejects_credentialed_urls(self):
        for value in (
            "https://key@provider.test/v1",
            "https://provider.test/v1?token=secret",
            "file:///tmp/a",
        ):
            with self.assertRaises(ValueError):
                endpoint(value)

    def check_physical_deadline(self, dribble_headers):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if dribble_headers:
                    try:
                        self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Dribble: ")
                        self.wfile.flush()
                        for _ in range(100):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                            time.sleep(0.03)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(TransportError):
                exchange(f"http://127.0.0.1:{server.server_port}", "synthetic", {}, timeout=0.15)
            self.assertLess(time.monotonic() - started, 0.7)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_physical_deadline_stops_trickling_response(self):
        self.check_physical_deadline(False)

    def test_physical_deadline_stops_trickling_headers(self):
        self.check_physical_deadline(True)

    def test_http_redirect_never_forwards_credentials(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                requests.append(self.path)
                self.send_response(307)
                self.send_header("Location", "/credential-sink")
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with self.assertRaisesRegex(TransportError, "http_307"):
                exchange(f"http://127.0.0.1:{server.server_port}/start", "secret", {})
            self.assertEqual(requests, ["/start"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
