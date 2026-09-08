"""Bounded MCP lease and lifecycle-telemetry heartbeat contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from penelopa_runtime.broker import Broker, TransportError, exchange
from penelopa_runtime.config import Settings
from penelopa_runtime.state import RuntimeState


def settings(root: str, **overrides) -> Settings:
    home = Path(root) / "hermes"
    home.mkdir()
    values = {
        "task_id": "00000000-0000-0000-0000-000000000001",
        "user_id": "00000000-0000-0000-0000-000000000002",
        "claim_version": 1,
        "api_base": "http://backend.invalid",
        "provider_base": "http://provider.invalid/v1",
        "model": "primary",
        "fallback_models": ("fallback",),
        "task_token": "test-task-secret",
        "lifecycle_token": "test-lifecycle-secret",
        "provider_token": "test-provider-secret",
        "home": home,
    }
    values.update(overrides)
    return Settings(**values)


def lease_response(config: Settings) -> dict:
    return {
        "result": {
            "structuredContent": {
                "task_id": config.task_id,
                "status": "RUNNING",
                "lease_expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=180)
                ).isoformat(),
            }
        }
    }


class HeartbeatContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = settings(self.temp.name)
        self.state = RuntimeState(self.settings)
        self.broker = Broker(self.settings, self.state)
        self.addCleanup(self.broker.server.server_close)

    @staticmethod
    def operation_events(state: RuntimeState, operation: str) -> list[dict]:
        return [
            event
            for event in state.snapshot()["diagnostic_events"]
            if event.get("operation") == operation
        ]

    def test_mcp_lease_and_lifecycle_telemetry_are_separate_operations(self):
        with patch.object(
            self.broker, "remote_rpc", return_value=lease_response(self.settings)
        ):
            self.assertTrue(self.broker._renew_mcp_lease())
        with patch.object(
            self.broker, "lifecycle", return_value={"status": "RUNNING"}
        ) as lifecycle:
            self.broker.heartbeat()

        telemetry_payload = lifecycle.call_args.args[1]
        self.assertIn("terminal_accepted", telemetry_payload)
        self.assertIn("provider_model", telemetry_payload)
        self.assertIn("diagnostic_events", telemetry_payload)

        mcp_events = self.operation_events(self.state, "mcp_lease_heartbeat")
        lifecycle_events = self.operation_events(self.state, "lifecycle_telemetry")
        self.assertEqual([event["status"] for event in mcp_events], ["started", "completed"])
        self.assertEqual([event["count"] for event in mcp_events], [1, 1])
        self.assertTrue(all(event["mcp_tool"] == "heartbeat_task" for event in mcp_events))
        self.assertEqual(
            [event["status"] for event in lifecycle_events], ["started", "completed"]
        )
        self.assertEqual([event["count"] for event in lifecycle_events], [1, 1])
        self.assertTrue(all("mcp_tool" not in event for event in lifecycle_events))

    def test_lifecycle_telemetry_failure_is_visible_but_does_not_stop_valid_lease(self):
        with patch.object(
            self.broker,
            "lifecycle",
            side_effect=TransportError("http_503", http_status=503),
        ):
            with self.assertRaises(TransportError):
                self.broker.heartbeat()

        events = self.operation_events(self.state, "lifecycle_telemetry")
        self.assertEqual([event["status"] for event in events], ["started", "failed"])
        self.assertEqual(events[-1]["error_code"], "http_503")
        self.assertEqual(events[-1]["http_status"], 503)
        self.assertTrue(events[-1]["retryable"])
        self.assertFalse(self.state.stop.is_set())
        self.assertNotIn("mcp_lease_heartbeat", self.state.operation_counts)

    def test_heartbeat_loop_keeps_fresh_mcp_lease_when_lifecycle_telemetry_fails(self):
        config = replace(self.settings, heartbeat_interval=0.001)
        state = RuntimeState(config)
        broker = Broker(config, state)
        self.addCleanup(broker.server.server_close)

        def lifecycle_failure(*_args, **_kwargs):
            broker.done.set()
            raise TransportError("http_503", http_status=503)

        with (
            patch.object(broker, "remote_rpc", return_value=lease_response(config)),
            patch.object(broker, "lifecycle", side_effect=lifecycle_failure),
        ):
            broker.heartbeat_loop()

        self.assertFalse(state.stop.is_set())
        self.assertEqual(
            [event["status"] for event in self.operation_events(state, "mcp_lease_heartbeat")],
            ["started", "completed"],
        )
        self.assertEqual(
            [event["status"] for event in self.operation_events(state, "lifecycle_telemetry")],
            ["started", "failed"],
        )

    def test_mcp_lease_failure_stops_after_its_own_three_failures(self):
        config = replace(self.settings, heartbeat_interval=0.001)
        state = RuntimeState(config)
        broker = Broker(config, state)
        self.addCleanup(broker.server.server_close)
        with (
            patch.object(
                broker,
                "remote_rpc",
                side_effect=TransportError("http_503", http_status=503),
            ),
            patch.object(broker, "lifecycle") as lifecycle,
        ):
            broker.heartbeat_loop()

        self.assertTrue(state.stop.is_set())
        self.assertEqual(state.error_code, "mcp_lease_heartbeat_unavailable")
        lifecycle.assert_not_called()
        events = self.operation_events(state, "mcp_lease_heartbeat")
        self.assertEqual(
            [(event["status"], event["count"]) for event in events],
            [
                ("started", 1),
                ("failed", 1),
                ("started", 2),
                ("failed", 2),
                ("started", 3),
                ("failed", 3),
            ],
        )
        self.assertTrue(all(event["retryable"] for event in events if event["status"] == "failed"))

    def test_mcp_heartbeat_cap_survives_same_claim_restart(self):
        config = replace(self.settings, mcp_max_heartbeats=1)
        state = RuntimeState(config)
        broker = Broker(config, state)
        self.addCleanup(broker.server.server_close)
        with patch.object(broker, "remote_rpc", return_value=lease_response(config)):
            self.assertTrue(broker._renew_mcp_lease())

        restored_state = RuntimeState(config)
        restored = Broker(config, restored_state)
        self.addCleanup(restored.server.server_close)
        self.assertFalse(restored._renew_mcp_lease())

        self.assertTrue(restored_state.stop.is_set())
        self.assertEqual(restored_state.error_code, "mcp_heartbeat_limit_exhausted")
        events = self.operation_events(restored_state, "mcp_lease_heartbeat")
        self.assertEqual(events[-1]["status"], "rejected")
        self.assertEqual(events[-1]["count"], 2)
        self.assertEqual(events[-1]["error_code"], "mcp_heartbeat_limit_exhausted")
        self.assertFalse(events[-1]["retryable"])

    def test_report_failure_marks_heartbeat_cap_non_retryable(self):
        with (
            patch.object(self.broker, "receipt", return_value=None),
            patch.object(self.broker, "remote_rpc") as remote_rpc,
        ):
            self.broker.report_failure("mcp_heartbeat_limit_exhausted")
        arguments = remote_rpc.call_args.args[0]["params"]["arguments"]
        self.assertFalse(arguments["retryable"])
        self.assertEqual(
            set(arguments["details"]),
            {"runtime_phase", "diagnostic_events", "terminal_state"},
        )

    def test_exchange_retains_only_safe_http_facts_from_transport(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "error": "http_422",
                    "http_status": 422,
                    "provider_request_id": "req_safe_123",
                }
            ).encode(),
        )
        with patch("penelopa_runtime.broker.subprocess.run", return_value=completed):
            with self.assertRaises(TransportError) as caught:
                exchange("http://provider.invalid", "ignored", {})
        self.assertEqual(caught.exception.code, "http_422")
        self.assertEqual(caught.exception.http_status, 422)
        self.assertEqual(caught.exception.provider_request_id, "req_safe_123")


class HeartbeatConfigTest(unittest.TestCase):
    def test_mcp_heartbeat_limit_is_loaded_and_validated(self):
        environment = {
            "HERMES_TASK_ID": "00000000-0000-0000-0000-000000000001",
            "HERMES_USER_ID": "00000000-0000-0000-0000-000000000002",
            "HERMES_CLAIM_VERSION": "1",
            "HERMES_INTERNAL_API_BASE_URL": "http://backend.invalid",
            "HERMES_BASE_URL": "http://provider.invalid/v1",
            "HERMES_MODEL": "primary",
            "HERMES_FALLBACK_MODELS": "[]",
            "HERMES_TASK_TOKEN": "task-secret",
            "HERMES_LIFECYCLE_TOKEN": "lifecycle-secret",
            "HERMES_API_TOKEN": "provider-secret",
            "HERMES_MCP_MAX_HEARTBEATS": "7",
        }
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(Settings.from_env().mcp_max_heartbeats, 7)
        environment["HERMES_MCP_MAX_HEARTBEATS"] = "0"
        with patch.dict("os.environ", environment, clear=True):
            with self.assertRaisesRegex(ValueError, "heartbeat limit"):
                Settings.from_env()
