"""Loopback transport boundary: secrets and lifecycle calls never enter the agent."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from penelopa_runtime import MCP_TOOLS
from penelopa_runtime.diagnostics import provider_request_id_from_headers
from penelopa_runtime.state import atomic_json

_HEARTBEAT_FAILURE_LIMIT = 3
_NON_RETRYABLE_FAILURE_CODES = frozenset(
    {
        "provider_request_rejected",
        "budget_exhausted",
        "mcp_heartbeat_limit_exhausted",
    }
)


class CapabilityRevoked(RuntimeError):
    pass


class TransportError(RuntimeError):
    """A bounded transport classification; never carry a response body."""

    def __init__(
        self,
        code,
        *,
        http_status=None,
        provider_request_id=None,
    ):
        super().__init__(str(code))
        self.code = str(code)
        self.http_status = (
            http_status
            if type(http_status) is int and 100 <= http_status <= 599
            else None
        )
        # The transport subprocess emits this only from a fixed header allowlist.
        # Keep it opaque here; the diagnostics boundary validates it again.
        self.provider_request_id = (
            provider_request_id if isinstance(provider_request_id, str) else None
        )


def exchange(url, token, payload=None, *, timeout=30, method="POST", headers=None):
    # A socket-inactivity timeout cannot bound DNS or trickling headers/SSE.
    # subprocess.run kills AND reaps this fixed worker on deadline expiry.
    # Pipes keep secrets out of argv, environment, persistent files and logs.
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "penelopa_runtime.transport"],
            input=json.dumps(
                {
                    "url": url,
                    "token": token,
                    "payload": payload,
                    "timeout": timeout,
                    "method": method,
                    "headers": headers,
                }
            ).encode(),
            capture_output=True,
            timeout=timeout,
            check=False,
            env={
                "PATH": os.defpath,
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        if completed.returncode:
            raise TransportError("transport_unavailable")
        result = json.loads(completed.stdout)
        if result.get("error"):
            raise TransportError(
                result["error"],
                http_status=result.get("http_status"),
                provider_request_id=result.get("provider_request_id"),
            )
        return result["status"], result["headers"], base64.b64decode(result["body"], validate=True)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        raise TransportError("transport_unavailable") from None


def decode_rpc(raw: bytes):
    if not raw.strip():
        return None
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except ValueError:
        for line in text.splitlines():
            if line.startswith("data:"):
                try:
                    value = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(value, dict) and ("result" in value or "error" in value):
                    return value
    raise TransportError("invalid_rpc_response")


def structured(result):
    if not isinstance(result, dict):
        return {}
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for part in result.get("content", []):
        if isinstance(part, dict) and part.get("type") == "text":
            try:
                value = json.loads(part.get("text", ""))
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def rpc_error(request_id, message, code=-32602):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class Broker:
    def __init__(self, settings, state):
        self.settings, self.state = settings, state
        self.terminal_lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.server_thread = threading.Thread(target=self.server.serve_forever, name="broker")
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, name="heartbeat")
        self.lease_thread = threading.Thread(target=self.lease_watchdog, name="lease-watchdog")
        self.lease_ready = threading.Event()
        self.lease_deadline = time.monotonic() + min(settings.internal_timeout, 30)
        self.done = threading.Event()
        self.pending_path = settings.home.parent / "pending_terminal.json"
        self.task_kind = "recommendation_generation"
        self.budget = None
        self.budget_lock = threading.RLock()
        self.budget_deadline = None
        self.reserved = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def configure_task(self, brief):
        self.task_kind = brief.get("task_kind", "recommendation_generation")
        if self.task_kind != "self_improvement":
            return
        self.budget = brief["budget"]
        deadline = datetime.fromisoformat(brief["deadline"])
        self.budget_deadline = time.monotonic() + max(0, (deadline - datetime.now(timezone.utc)).total_seconds())
        path = self.settings.home / "penelopa-review-budget.json"
        if path.is_symlink():
            raise ValueError("budget_checkpoint_symlink")
        old = json.loads(path.read_text()) if path.exists() else {}
        if old.get("task_id") == self.settings.task_id and old.get("claim_version") == self.settings.claim_version:
            self.reserved = old["reserved"]

    def reserve_call(self, payload):
        if self.budget is None:
            return payload
        # UTF-8 bytes conservatively bound tokenized input; reserve output before
        # dispatch so transport failures and missing usage cannot reopen budget.
        input_size = len(json.dumps(payload, ensure_ascii=False).encode()) + 1024
        output = min(2048, int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 2048),
            self.budget["output_tokens"] - self.reserved["output_tokens"])
        if (time.monotonic() >= self.budget_deadline or output < 1 or
            self.reserved["calls"] >= self.budget["calls"] or
            self.reserved["input_tokens"] + input_size > self.budget["input_tokens"]):
            self.state.checkpoint(error_code="budget_exhausted")
            self.state.stop.set()
            raise TransportError("budget_exhausted")
        self.reserved = {"calls": self.reserved["calls"] + 1,
            "input_tokens": self.reserved["input_tokens"] + input_size,
            "output_tokens": self.reserved["output_tokens"] + output}
        atomic_json(self.settings.home / "penelopa-review-budget.json", {"task_id": self.settings.task_id,
            "claim_version": self.settings.claim_version, "reserved": self.reserved})
        result = {**payload, "stream": False}
        result.pop("stream_options", None)
        result.pop("max_completion_tokens", None)
        result["max_tokens"] = output
        return result

    def start(self):
        self.server_thread.start()
        self.heartbeat_thread.start()
        self.lease_thread.start()

    def close(self):
        self.done.set()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.heartbeat_thread.join(timeout=self.settings.internal_timeout + 1)
        self.lease_thread.join()

    def lifecycle(self, suffix, payload=None, method="POST", timeout=None):
        try:
            _, _, raw = exchange(
                f"{self.settings.api_base}/api/internal/hermes/tasks/{self.settings.task_id}/{suffix}",
                self.settings.lifecycle_token,
                payload,
                timeout=timeout or self.settings.internal_timeout,
                method=method,
            )
        except TransportError as error:
            if error.code in {"http_401", "http_403", "http_404", "http_409"}:
                self.state.stop.set()
                raise CapabilityRevoked("claim_capability_revoked") from None
            raise
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise TransportError("invalid_lifecycle_response")
        return result

    def _terminal_state(self):
        if self.state.accepted is not None:
            return "accepted"
        if self.state.phase == "terminal_pending":
            return "pending"
        return "not_started"

    def _record_terminal_event(
        self,
        status,
        *,
        operation,
        count,
        terminal_state=None,
        mcp_tool=None,
        started_at=None,
        error_code=None,
        http_status=None,
        retryable=None,
    ):
        fields = {"operation": operation, "count": count}
        if terminal_state is not None:
            fields["terminal_state"] = terminal_state
        if mcp_tool is not None:
            fields["mcp_tool"] = mcp_tool
        if started_at is not None:
            fields["duration_ms"] = max(0, int((time.monotonic() - started_at) * 1000))
        if error_code is not None:
            fields["error_code"] = error_code
        if http_status is not None:
            fields["http_status"] = http_status
        if retryable is not None:
            fields["retryable"] = retryable
        self.state.record_diagnostic_event("terminal", status, **fields)

    def receipt(self):
        count = self.state.next_operation_count("terminal_receipt")
        started_at = time.monotonic()
        self._record_terminal_event(
            "started",
            operation="terminal_receipt",
            count=count,
            terminal_state=self._terminal_state(),
        )
        try:
            result = self.lifecycle("receipt", method="GET")
            if (
                str(result.get("task_id")) != self.settings.task_id
                or int(result.get("claim_version", -1)) != self.settings.claim_version
            ):
                self.state.stop.set()
                raise CapabilityRevoked("receipt_claim_mismatch")
            if result.get("accepted") is True and result.get("status") == "SUCCEEDED":
                receipt = result.get("receipt")
                if not isinstance(receipt, dict):
                    raise TransportError("invalid_terminal_receipt")
                self.state.checkpoint("terminal_accepted", accepted=receipt)
                atomic_json(
                    self.pending_path,
                    {
                        "task_id": self.settings.task_id,
                        "claim_version": self.settings.claim_version,
                        "accepted": True,
                    },
                )
                self._record_terminal_event(
                    "completed",
                    operation="terminal_receipt",
                    count=count,
                    terminal_state="accepted",
                    started_at=started_at,
                )
                return receipt
            if result.get("status") in {"FAILED", "CANCELLED"}:
                self.state.stop.set()
                raise CapabilityRevoked("task_no_longer_active")
            self._record_terminal_event(
                "completed",
                operation="terminal_receipt",
                count=count,
                terminal_state=self._terminal_state(),
                started_at=started_at,
            )
            return None
        except CapabilityRevoked:
            self._record_terminal_event(
                "rejected",
                operation="terminal_receipt",
                count=count,
                terminal_state="rejected",
                started_at=started_at,
                error_code="claim_capability_revoked",
                retryable=False,
            )
            raise
        except (TransportError, ValueError, TypeError) as error:
            if isinstance(error, TransportError):
                error_code, http_status = error.code, error.http_status
                retryable = self._retryable_transport_error(error_code)
            else:
                error_code, http_status, retryable = "terminal_receipt_invalid", None, False
            self._record_terminal_event(
                "failed",
                operation="terminal_receipt",
                count=count,
                terminal_state="receipt_unknown",
                started_at=started_at,
                error_code=error_code,
                http_status=http_status,
                retryable=retryable,
            )
            raise

    def await_initial_lease(self):
        while not self.lease_ready.wait(0.1):
            if self.state.accepted is not None:
                return
            if self.state.stop.is_set():
                raise CapabilityRevoked("initial_task_lease_unavailable")

    def update_task_lease(self, response):
        result = response.get("result", {}) if isinstance(response, dict) else {}
        data = structured(result)
        if str(data.get("task_id")) != self.settings.task_id or data.get("status") != "RUNNING":
            raise TransportError("invalid_task_heartbeat")
        try:
            expires = datetime.fromisoformat(data["lease_expires_at"].replace("Z", "+00:00"))
            if expires.tzinfo is None:
                raise ValueError("missing_timezone")
            remaining = (expires - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                raise ValueError("expired_lease")
        except (KeyError, ValueError, TypeError, AttributeError):
            raise TransportError("invalid_task_lease") from None
        # Monotonic after the server's absolute timestamp has been converted.
        self.lease_deadline = time.monotonic() + remaining
        self.lease_ready.set()

    def lease_watchdog(self):
        while not self.done.wait(0.1):
            if self.budget_deadline is not None and time.monotonic() >= self.budget_deadline:
                self.state.checkpoint(error_code="budget_exhausted")
                self.state.stop.set()
                return
            if self.state.accepted is None and time.monotonic() >= self.lease_deadline:
                self.state.checkpoint(error_code="task_lease_expired")
                self.state.stop.set()
                return

    @staticmethod
    def _retryable_transport_error(code):
        if code in {"transport_unavailable", "http_408", "http_429"}:
            return True
        if isinstance(code, str) and code.startswith("http_"):
            try:
                return 500 <= int(code[5:]) < 600
            except ValueError:
                return False
        return False

    def _record_heartbeat_event(
        self,
        status,
        *,
        operation,
        count,
        started_at=None,
        error_code=None,
        http_status=None,
        retryable=None,
    ):
        fields = {
            "operation": operation,
            "count": count,
        }
        if operation == "mcp_lease_heartbeat":
            fields["mcp_tool"] = "heartbeat_task"
        if started_at is not None:
            fields["duration_ms"] = max(0, int((time.monotonic() - started_at) * 1000))
        if error_code is not None:
            fields["error_code"] = error_code
        if http_status is not None:
            fields["http_status"] = http_status
        if retryable is not None:
            fields["retryable"] = retryable
        self.state.record_diagnostic_event("heartbeat", status, **fields)

    @staticmethod
    def _heartbeat_error_details(error, fallback):
        if isinstance(error, TransportError):
            return error.code, error.http_status
        return fallback, None

    def heartbeat(self):
        count = self.state.next_operation_count("lifecycle_telemetry")
        started_at = time.monotonic()
        self._record_heartbeat_event(
            "started",
            operation="lifecycle_telemetry",
            count=count,
        )
        snapshot = self.state.snapshot()
        try:
            result = self.lifecycle(
                "runtime-heartbeat",
                {
                    key: snapshot[key]
                    for key in (
                        "claim_version",
                        "runtime_phase",
                        "session_id",
                    "generation_stages",
                    "review_status",
                    "error_code",
                    "llm_requests",
                    "missing_usage_calls",
                    "terminal_accepted",
                    "provider_model",
                    "diagnostics_version",
                    "diagnostic_events",
                )
                },
                timeout=min(self.settings.internal_timeout, 30),
            )
        except CapabilityRevoked:
            self._record_heartbeat_event(
                "rejected",
                operation="lifecycle_telemetry",
                count=count,
                started_at=started_at,
                error_code="claim_capability_revoked",
                retryable=False,
            )
            raise
        except (TransportError, ValueError) as error:
            error_code, http_status = self._heartbeat_error_details(
                error, "lifecycle_telemetry_failed"
            )
            self._record_heartbeat_event(
                "failed",
                operation="lifecycle_telemetry",
                count=count,
                started_at=started_at,
                error_code=error_code,
                http_status=http_status,
                retryable=self._retryable_transport_error(error_code),
            )
            raise
        self._record_heartbeat_event(
            "completed",
            operation="lifecycle_telemetry",
            count=count,
            started_at=started_at,
        )
        return result

    def _renew_mcp_lease(self):
        """Renew the data/claim lease independently from lifecycle telemetry."""

        count = self.state.next_operation_count("mcp_lease_heartbeat")
        if count > self.settings.mcp_max_heartbeats:
            self._record_heartbeat_event(
                "rejected",
                operation="mcp_lease_heartbeat",
                count=count,
                error_code="mcp_heartbeat_limit_exhausted",
                retryable=False,
            )
            self.state.checkpoint(error_code="mcp_heartbeat_limit_exhausted")
            self.state.stop.set()
            return False

        started_at = time.monotonic()
        self._record_heartbeat_event(
            "started",
            operation="mcp_lease_heartbeat",
            count=count,
        )
        try:
            response = self.remote_rpc(
                {
                    "jsonrpc": "2.0",
                    "id": "heartbeat",
                    "method": "tools/call",
                    "params": {"name": "heartbeat_task", "arguments": {}},
                },
                timeout=min(self.settings.internal_timeout, 30),
            )
            if (
                not isinstance(response, dict)
                or "error" in response
                or not isinstance(response.get("result"), dict)
                or response["result"].get("isError")
            ):
                if self.receipt() is None:
                    raise TransportError("task_heartbeat_rejected")
            else:
                self.update_task_lease(response)
        except CapabilityRevoked:
            self._record_heartbeat_event(
                "rejected",
                operation="mcp_lease_heartbeat",
                count=count,
                started_at=started_at,
                error_code="claim_capability_revoked",
                retryable=False,
            )
            raise
        except (TransportError, ValueError) as error:
            # A revoked data capability can be expected after a terminal ACK.
            accepted_after_revoke = False
            if (
                isinstance(error, TransportError)
                and error.code in {"http_401", "http_403"}
            ):
                try:
                    accepted_after_revoke = self.receipt() is not None
                except CapabilityRevoked:
                    self._record_heartbeat_event(
                        "rejected",
                        operation="mcp_lease_heartbeat",
                        count=count,
                        started_at=started_at,
                        error_code="claim_capability_revoked",
                        retryable=False,
                    )
                    raise
            if accepted_after_revoke:
                self._record_heartbeat_event(
                    "rejected",
                    operation="mcp_lease_heartbeat",
                    count=count,
                    started_at=started_at,
                    error_code=error.code,
                    http_status=error.http_status,
                    retryable=False,
                )
                return True
            error_code, http_status = self._heartbeat_error_details(
                error, "mcp_lease_heartbeat_failed"
            )
            self._record_heartbeat_event(
                "failed",
                operation="mcp_lease_heartbeat",
                count=count,
                started_at=started_at,
                error_code=error_code,
                http_status=http_status,
                retryable=self._retryable_transport_error(error_code),
            )
            raise
        self._record_heartbeat_event(
            "completed",
            operation="mcp_lease_heartbeat",
            count=count,
            started_at=started_at,
        )
        return True

    def heartbeat_loop(self):
        mcp_lease_failures = 0
        while not self.done.is_set():
            if self.state.accepted is None:
                try:
                    if not self._renew_mcp_lease():
                        return
                except CapabilityRevoked:
                    return
                except (TransportError, ValueError):
                    mcp_lease_failures += 1
                    if mcp_lease_failures >= _HEARTBEAT_FAILURE_LIMIT:
                        self.state.checkpoint(error_code="mcp_lease_heartbeat_unavailable")
                        self.state.stop.set()
                        return
                    self.done.wait(self.settings.heartbeat_interval)
                    continue
                mcp_lease_failures = 0
            try:
                # Lifecycle telemetry is diagnostic-only; a fresh MCP lease must
                # not be killed merely because the telemetry endpoint is down.
                self.heartbeat()
            except CapabilityRevoked:
                return
            except (TransportError, ValueError):
                pass
            self.done.wait(self.settings.heartbeat_interval)

    def remote_rpc(self, message, timeout=None):
        _, _, raw = exchange(
            f"{self.settings.api_base}/_internal/hermes/mcp",
            self.settings.task_token,
            message,
            timeout=timeout or self.settings.internal_timeout,
            headers={"MCP-Protocol-Version": "2025-03-26"},
        )
        return decode_rpc(raw)

    def call(self, name, arguments=None):
        response = self.mcp(
            {
                "jsonrpc": "2.0",
                "id": "supervisor",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        if not isinstance(response, dict) or "error" in response:
            raise TransportError("mcp_call_failed")
        result = response.get("result", {})
        if result.get("isError"):
            raise TransportError("mcp_tool_error")
        return structured(result)

    def mcp(self, message):
        if not isinstance(message, dict):
            return rpc_error(None, "Object request required")
        request_id = message.get("id")
        method = message.get("method")
        # Fully local handshake: no sampling/elicitation/resources capabilities.
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": message.get("params", {}).get(
                        "protocolVersion", "2025-03-26"
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "penelopa-scoped-bridge", "version": "0.1.0"},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "notifications/initialized":
            return None
        if method not in {"tools/list", "tools/call"}:
            return rpc_error(request_id, "Capability is not available", -32601)
        if self.state.stop.is_set():
            return rpc_error(request_id, "Task interrupted")
        if method == "tools/list":
            response = self.remote_rpc(message)
            if not isinstance(response, dict) or "error" in response:
                return rpc_error(request_id, "Tool discovery unavailable")
            tools = response.get("result", {}).get("tools", [])
            response["result"] = {
                "tools": [tool for tool in tools if tool.get("name") in self.allowed_tools()]
            }
            return response
        params = message.get("params", {})
        name = params.get("name")
        if name not in self.allowed_tools():
            return rpc_error(request_id, "Tool is not permitted")
        if self.state.accepted is not None:
            if name in {"submit_recommendations", "complete_self_improvement"}:
                return self.cached_terminal(request_id)
            return rpc_error(request_id, "Task already accepted; remote tools are closed")
        # The model cannot forge runner-owned telemetry/provenance.
        outgoing = {**message, "params": {"name": name, "arguments": params.get("arguments", {})}}
        if name not in {"submit_recommendations", "complete_self_improvement"}:
            return self.remote_rpc(outgoing)
        with self.terminal_lock:
            if self.state.accepted is not None:
                return self.cached_terminal(request_id)
            count = self.state.next_operation_count("terminal_submission")
            started_at = time.monotonic()
            self._record_terminal_event(
                "started",
                operation="terminal_submission",
                count=count,
                terminal_state="pending",
                mcp_tool=name,
            )
            self.state.checkpoint("terminal_pending")
            atomic_json(
                self.pending_path,
                {
                    "task_id": self.settings.task_id,
                    "claim_version": self.settings.claim_version,
                    "payload": outgoing["params"]["arguments"],
                },
            )
            outgoing["params"]["_meta"] = self.state.terminal_meta()
            try:
                response = self.remote_rpc(outgoing)
            except TransportError as error:
                # A timed-out terminal POST may already be committed. Never replay
                # it until claim-bound receipt readback has established the state.
                if self.receipt() is not None:
                    self._record_terminal_event(
                        "completed",
                        operation="terminal_submission",
                        count=count,
                        terminal_state="accepted",
                        mcp_tool=name,
                        started_at=started_at,
                    )
                    return self.cached_terminal(request_id)
                self.state.checkpoint("working")
                self._record_terminal_event(
                    "timeout",
                    operation="terminal_submission",
                    count=count,
                    terminal_state="receipt_unknown",
                    mcp_tool=name,
                    started_at=started_at,
                    error_code=error.code,
                    http_status=error.http_status,
                    retryable=self._retryable_transport_error(error.code),
                )
                return rpc_error(
                    request_id, "Terminal receipt not accepted; revalidate before retry"
                )
            if (
                isinstance(response, dict)
                and "error" not in response
                and not response.get("result", {}).get("isError")
            ):
                # The receipt endpoint, not generated text, is authoritative.
                if self.receipt() is not None:
                    self._record_terminal_event(
                        "completed",
                        operation="terminal_submission",
                        count=count,
                        terminal_state="accepted",
                        mcp_tool=name,
                        started_at=started_at,
                    )
                    return response
                self.state.checkpoint("working")
                self._record_terminal_event(
                    "rejected",
                    operation="terminal_submission",
                    count=count,
                    terminal_state="receipt_unknown",
                    mcp_tool=name,
                    started_at=started_at,
                    error_code="terminal_receipt_unconfirmed",
                    retryable=True,
                )
                return rpc_error(request_id, "Terminal receipt has not been confirmed")
            self.state.checkpoint("working")
            self._record_terminal_event(
                "rejected",
                operation="terminal_submission",
                count=count,
                terminal_state="rejected",
                mcp_tool=name,
                started_at=started_at,
                error_code="terminal_submission_rejected",
                retryable=True,
            )
            return response

    def cached_terminal(self, request_id):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "isError": False,
                "structuredContent": self.state.accepted,
                "content": [{"type": "text", "text": json.dumps(self.state.accepted)}],
            },
        }

    def allowed_tools(self):
        if self.task_kind == "self_improvement":
            return {"get_task_brief", "complete_self_improvement"}
        return MCP_TOOLS - {"complete_self_improvement"}

    def provider(self, path, payload):
        with self.budget_lock:
            return self._provider(path, payload)

    def _record_provider_event(
        self,
        status,
        *,
        stage,
        model,
        count,
        started_at=None,
        error_code=None,
        http_status=None,
        provider_request_id=None,
        retryable=None,
    ):
        fields = {
            "stage": stage,
            "model": model,
            "operation": "provider_request",
            "count": count,
        }
        if started_at is not None:
            fields["duration_ms"] = max(0, int((time.monotonic() - started_at) * 1000))
        if error_code is not None:
            fields["error_code"] = error_code
        if http_status is not None:
            fields["http_status"] = http_status
        if provider_request_id is not None:
            fields["provider_request_id"] = provider_request_id
        if retryable is not None:
            fields["retryable"] = retryable
        self.state.record_diagnostic_event("provider", status, **fields)

    def _provider(self, path, payload):
        if self.state.stop.is_set():
            raise TransportError("task_interrupted")
        stage = path.split("/")[2]
        if stage not in {
            "analysis",
            "goal_check",
            "compaction",
            "session_search",
            "background_review",
        }:
            raise TransportError("provider_route_denied")
        if self.state.phase == "reviewing":
            stage = "background_review"
        if payload.get("model") not in {self.settings.model, *self.settings.fallback_models}:
            raise TransportError("provider_model_denied")
        models = [payload["model"], *self.settings.fallback_models]
        for model in dict.fromkeys(models):
            count = self.state.next_operation_count("provider_request")
            started = time.monotonic()
            self._record_provider_event(
                "started", stage=stage, model=model, count=count
            )
            reserved_before = dict(self.reserved)
            try:
                outgoing = self.reserve_call({**payload, "model": model})
            except TransportError as error:
                self._record_provider_event(
                    "rejected",
                    stage=stage,
                    model=model,
                    count=count,
                    started_at=started,
                    error_code=error.code,
                    http_status=error.http_status,
                    provider_request_id=error.provider_request_id,
                    retryable=False,
                )
                raise
            try:
                http_status, headers, raw = exchange(
                    f"{self.settings.provider_base}/chat/completions",
                    self.settings.provider_token,
                    outgoing,
                    timeout=min(self.settings.provider_timeout, max(0.1, self.budget_deadline - time.monotonic())) if self.budget_deadline else self.settings.provider_timeout,
                    headers={"X-Penelopa-Stage": stage},
                )
            except TransportError as error:
                retryable = self._retryable_transport_error(error.code)
                # Establish the durable task classification in memory before an
                # aggregate usage checkpoint can fail. The diagnostic event is
                # itself best-effort, so a volume I/O fault cannot replace a
                # provider 4xx with an opaque OSError during outer unwinding.
                if not retryable:
                    self.state.error_code = "provider_request_rejected"
                self._record_provider_event(
                    "failed" if retryable else "rejected",
                    stage=stage,
                    model=model,
                    count=count,
                    started_at=started,
                    error_code=error.code,
                    http_status=error.http_status,
                    provider_request_id=error.provider_request_id,
                    retryable=retryable,
                )
                try:
                    self.state.record(stage, model, time.monotonic() - started, error=error.code)
                except OSError:
                    # Usage aggregation is diagnostic accounting. It must not
                    # mask the original provider classification on a degraded
                    # state volume after the safe event has been captured.
                    pass
                if not retryable:
                    self.state.stop.set()
                    raise
                continue
            usage = None
            try:
                if raw.lstrip().startswith(b"{"):
                    usage = json.loads(raw).get("usage")
                else:
                    for line in raw.decode().splitlines():
                        if line.startswith("data:") and line[5:].strip() != "[DONE]":
                            usage = json.loads(line[5:]).get("usage") or usage
            except (ValueError, AttributeError):
                pass
            self._record_provider_event(
                "completed",
                stage=stage,
                model=model,
                count=count,
                started_at=started,
                http_status=http_status,
                provider_request_id=provider_request_id_from_headers(headers),
            )
            try:
                self.state.record(stage, model, time.monotonic() - started, usage=usage)
            except OSError:
                # The provider response remains valid even if optional
                # aggregate accounting cannot be flushed right now.
                pass
            if self.budget and isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                reported_input = usage["prompt_tokens"]
                total = usage.get("total_tokens")
                inconsistent = type(total) is int and total != reported_input + usage["completion_tokens"]
                if inconsistent:
                    # Some providers exclude cached input from prompt_tokens.
                    reported_input = max(reported_input, total - usage["completion_tokens"])
                reconciled = reserved_before["input_tokens"] + reported_input
                self.reserved["input_tokens"] = max(self.reserved["input_tokens"], reconciled) if inconsistent else reconciled
                self.reserved["output_tokens"] = reserved_before["output_tokens"] + usage["completion_tokens"]
                atomic_json(self.settings.home / "penelopa-review-budget.json", {"task_id": self.settings.task_id,
                    "claim_version": self.settings.claim_version, "reserved": self.reserved})
                if any(self.reserved[key] >= self.budget[key] for key in ("input_tokens", "output_tokens")):
                    self.state.checkpoint(error_code="budget_exhausted")
                    self.state.stop.set()
            return headers.get("Content-Type", "application/json"), raw
        self.state.checkpoint(error_code="provider_chain_unavailable")
        self.state.stop.set()
        raise TransportError("provider_chain_unavailable")

    def report_failure(self, error_code):
        if self.state.accepted is not None:
            return
        # Resolve an ambiguous terminal write before attempting ANY other
        # terminal operation. This capability stays separate from model tools.
        if self.receipt() is not None:
            return
        snapshot = self.state.snapshot()
        if snapshot.get("terminal_accepted"):
            terminal_state = "accepted"
        elif snapshot.get("runtime_phase") == "terminal_pending":
            terminal_state = "pending"
        else:
            terminal_state = "not_started"
        self.remote_rpc(
            {
                "jsonrpc": "2.0",
                "id": "supervisor-failure",
                "method": "tools/call",
                "params": {
                    "name": "report_failure",
                    "arguments": {
                        "error_type": "hermes_runtime_error",
                        "error_code": error_code[:128],
                        "message": "Managed Hermes runtime stopped before terminal acceptance.",
                        "retryable": error_code not in _NON_RETRYABLE_FAILURE_CODES,
                        "details": {
                            "runtime_phase": snapshot["runtime_phase"],
                            "diagnostic_events": snapshot["diagnostic_events"],
                            "terminal_state": terminal_state,
                        },
                    },
                },
            }
        )

    def handler(self):
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError("request_size")
                    payload = json.loads(self.rfile.read(length))
                    if self.path == "/mcp":
                        value = broker.mcp(payload)
                        raw = b"" if value is None else json.dumps(value).encode()
                        content_type = "application/json"
                    elif self.path.startswith("/provider/") and self.path.endswith(
                        "/v1/chat/completions"
                    ):
                        content_type, raw = broker.provider(self.path, payload)
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200 if raw else 204)
                except CapabilityRevoked:
                    content_type, raw = (
                        "application/json",
                        b'{"error":{"message":"task_capability_revoked"}}',
                    )
                    self.send_response(403)
                except (ValueError, TransportError):
                    content_type, raw = (
                        "application/json",
                        b'{"error":{"message":"transport_request_failed"}}',
                    )
                    self.send_response(502)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler
