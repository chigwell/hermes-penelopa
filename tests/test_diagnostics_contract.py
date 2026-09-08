"""Content-free native runtime diagnostics unit tests."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from penelopa_runtime import transport
from penelopa_runtime.broker import Broker, TransportError
from penelopa_runtime.config import Settings
from penelopa_runtime.diagnostics import MAX_EVENTS, provider_request_id_from_headers
from penelopa_runtime.state import RuntimeState


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


class DiagnosticsContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = settings(self.temp.name)
        self.state = RuntimeState(self.settings)

    def test_event_contract_drops_untrusted_fields_and_never_persists_credentials(self):
        event = self.state.record_diagnostic_event(
            "provider",
            "rejected",
            stage="analysis",
            model="primary",
            duration_ms=17,
            error_code="http_422",
            http_status=422,
            provider_request_id="req_123:abc",
            retryable=False,
            operation="provider_request",
            count=1,
            message="provider response must not be retained",
            provider_url="https://provider.invalid/v1",
            secret="sk-never-persist-this",
        )
        self.assertEqual(
            event,
            {
                "sequence": 0,
                "kind": "provider",
                "status": "rejected",
                "stage": "analysis",
                "model": "primary",
                "duration_ms": 17,
                "error_code": "http_422",
                "http_status": 422,
                "provider_request_id": "req_123:abc",
                "retryable": False,
                "operation": "provider_request",
                "count": 1,
            },
        )
        persisted = self.state.path.read_text()
        self.assertNotIn("provider response", persisted)
        self.assertNotIn("provider.invalid", persisted)
        self.assertNotIn("sk-never-persist-this", persisted)

    def test_ring_and_operation_counts_survive_restart_without_reusing_sequences(self):
        for _ in range(MAX_EVENTS + 3):
            count = self.state.next_operation_count("provider_request")
            self.state.record_diagnostic_event(
                "provider", "started", operation="provider_request", count=count
            )
        snapshot = self.state.snapshot()
        self.assertEqual(len(snapshot["diagnostic_events"]), MAX_EVENTS)
        self.assertEqual(snapshot["diagnostic_events"][0]["sequence"], 3)
        self.assertEqual(snapshot["operation_counts"], {"provider_request": MAX_EVENTS + 3})
        restored = RuntimeState(self.settings)
        count = restored.next_operation_count("provider_request")
        event = restored.record_diagnostic_event(
            "provider", "completed", operation="provider_request", count=count
        )
        self.assertEqual(count, MAX_EVENTS + 4)
        self.assertEqual(event["sequence"], MAX_EVENTS + 3)

    def test_allowlisted_request_id_rejects_credentials_urls_and_unknown_headers(self):
        self.assertEqual(
            provider_request_id_from_headers({"X-Request-Id": "req_123:abc"}), "req_123:abc"
        )
        self.assertIsNone(provider_request_id_from_headers({"X-Request-Id": "Bearer secret"}))
        self.assertIsNone(provider_request_id_from_headers({"X-Request-Id": "https://secret.invalid"}))
        self.assertIsNone(provider_request_id_from_headers({"X-Trace": "req_123"}))

    def test_malformed_checkpoint_event_fails_closed_without_type_error(self):
        self.state.path.write_text(
            json.dumps(
                {
                    "task_id": self.settings.task_id,
                    "user_id": self.settings.user_id,
                    "claim_version": self.settings.claim_version,
                    "generation_stages": [],
                    "llm_requests": 0,
                    "missing_usage_calls": 0,
                    "diagnostic_events": [
                        {"sequence": 0, "kind": [], "status": "started"}
                    ],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "diagnostic checkpoint"):
            RuntimeState(self.settings)

    def test_malformed_checkpoint_metric_fails_closed_without_type_error(self):
        self.state.path.write_text(
            json.dumps(
                {
                    "task_id": self.settings.task_id,
                    "user_id": self.settings.user_id,
                    "claim_version": self.settings.claim_version,
                    "generation_stages": [
                        {"stage": [], "model": "primary", "duration_seconds": 1}
                    ],
                    "llm_requests": 0,
                    "missing_usage_calls": 0,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "metric checkpoint"):
            RuntimeState(self.settings)

    def test_diagnostic_checkpoint_write_is_best_effort(self):
        with patch("penelopa_runtime.state.atomic_json", side_effect=OSError("volume down")):
            event = self.state.record_diagnostic_event(
                "provider", "started", operation="provider_request", count=1
            )
        self.assertEqual(event["sequence"], 0)
        self.assertEqual(self.state.snapshot()["diagnostic_events"], [event])

    def test_http_error_emits_only_status_and_safe_request_id(self):
        headers = Message()
        headers["X-Request-Id"] = "req_123:abc"

        class Opener:
            def open(self, *_args, **_kwargs):
                raise urllib.error.HTTPError(
                    "https://provider.invalid/v1/chat/completions",
                    422,
                    "Unprocessable",
                    headers,
                    io.BytesIO(b'{"error":"secret provider body"}'),
                )

        stdin = io.StringIO(
            json.dumps(
                {
                    "url": "https://provider.invalid/v1/chat/completions",
                    "token": "provider-token",
                    "payload": {},
                    "timeout": 1,
                    "method": "POST",
                    "headers": {},
                }
            )
        )
        stdout = io.StringIO()
        with (
            patch("penelopa_runtime.transport.urllib.request.build_opener", return_value=Opener()),
            patch("sys.stdin", stdin),
            patch("sys.stdout", stdout),
        ):
            transport.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(
            result,
            {"error": "http_422", "http_status": 422, "provider_request_id": "req_123:abc"},
        )
        self.assertNotIn("secret provider body", stdout.getvalue())
        self.assertNotIn("provider-token", stdout.getvalue())

    def test_provider_rejection_preserves_safe_http_facts_and_terminal_policy(self):
        broker = Broker(self.settings, self.state)
        self.addCleanup(broker.server.server_close)
        with patch(
            "penelopa_runtime.broker.exchange",
            side_effect=TransportError(
                "http_422", http_status=422, provider_request_id="req_422:abc"
            ),
        ):
            with self.assertRaisesRegex(TransportError, "http_422"):
                broker.provider(
                    "/provider/analysis/v1/chat/completions",
                    {"model": "primary", "messages": [{"role": "user", "content": "private"}]},
                )
        events = self.state.snapshot()["diagnostic_events"]
        self.assertEqual([event["status"] for event in events], ["started", "rejected"])
        self.assertEqual(
            events[-1],
            {
                "sequence": 1,
                "kind": "provider",
                "status": "rejected",
                "stage": "analysis",
                "model": "primary",
                "duration_ms": events[-1]["duration_ms"],
                "error_code": "http_422",
                "http_status": 422,
                "provider_request_id": "req_422:abc",
                "retryable": False,
                "operation": "provider_request",
                "count": 1,
            },
        )
        self.assertEqual(self.state.error_code, "provider_request_rejected")
        self.assertNotIn("private", self.state.path.read_text())

    def test_provider_rejection_is_not_masked_by_metric_checkpoint_io_error(self):
        broker = Broker(self.settings, self.state)
        self.addCleanup(broker.server.server_close)
        with (
            patch(
                "penelopa_runtime.broker.exchange",
                side_effect=TransportError("http_422", http_status=422),
            ),
            patch("penelopa_runtime.state.atomic_json", side_effect=OSError("volume down")),
        ):
            with self.assertRaisesRegex(TransportError, "http_422"):
                broker.provider(
                    "/provider/analysis/v1/chat/completions", {"model": "primary"}
                )
        self.assertEqual(self.state.error_code, "provider_request_rejected")
        self.assertEqual(self.state.snapshot()["diagnostic_events"][-1]["error_code"], "http_422")

    def test_terminal_submission_emits_safe_pending_and_accepted_events(self):
        broker = Broker(self.settings, self.state)
        self.addCleanup(broker.server.server_close)
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "submit_recommendations",
                "arguments": {"private_model_text": "do not persist in diagnostics"},
            },
        }
        with (
            patch.object(broker, "remote_rpc", return_value={"result": {"isError": False}}),
            patch.object(broker, "receipt", return_value={"accepted": True}),
        ):
            result = broker.mcp(request)
        self.assertNotIn("error", result)
        events = self.state.snapshot()["diagnostic_events"]
        self.assertEqual([event["status"] for event in events], ["started", "completed"])
        self.assertEqual(events[0]["terminal_state"], "pending")
        self.assertEqual(events[0]["mcp_tool"], "submit_recommendations")
        self.assertEqual(events[1]["terminal_state"], "accepted")
        self.assertNotIn("private_model_text", str(events))
        self.assertNotIn("do not persist", str(events))


if __name__ == "__main__":
    unittest.main()
