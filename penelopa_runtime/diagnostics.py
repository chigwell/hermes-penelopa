"""Strict, content-free native runtime diagnostic event contract.

This module is deliberately dependency-free because it is used both by the
long-lived runtime and the short-lived transport helper.  Its output crosses a
container/volume boundary, so unknown fields are dropped instead of being
redacted after the fact.  Prompts, response bodies, URLs, exception messages,
arbitrary headers, and credentials are never valid diagnostic values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

VERSION = "native-runtime-diagnostics-v1"
MAX_EVENTS = 32

EVENT_KINDS = frozenset({"provider", "mcp", "terminal", "heartbeat", "runtime", "review"})
EVENT_STATUSES = frozenset({"started", "completed", "failed", "rejected", "timeout", "interrupted"})
STAGES = frozenset(
    {
        "search_friction",
        "search_opportunities",
        "terminal_recovery",
        "dedup",
        "analysis",
        "goal_check",
        "compaction",
        "session_search",
        "background_review",
    }
)
TERMINAL_STATES = frozenset({"not_started", "pending", "accepted", "rejected", "receipt_unknown"})
MCP_TOOLS = frozenset(
    {
        "submit_recommendations",
        "complete_self_improvement",
        "report_failure",
        "heartbeat_task",
    }
)
OPERATIONS = frozenset(
    {
        "provider_request",
        "mcp_lease_heartbeat",
        "lifecycle_telemetry",
        "terminal_submission",
        "terminal_receipt",
        "runtime_checkpoint",
        "review",
    }
)

SAFE_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}$")
CREDENTIAL_PREFIX = re.compile(
    r"(?i)^(?:bearer[-._:]|basic[-._:]|api[-_]?key[-._:]|authorization[-._:]|"
    r"sk-|pk-|xox[baprs]-|gh[pous]_|ya29\.|eyJ[A-Za-z0-9_-]{10})"
)
REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "openai-request-id",
    "x-openai-request-id",
    "anthropic-request-id",
    "x-amzn-requestid",
    "x-amz-request-id",
)


def _safe_code(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not SAFE_CODE.fullmatch(value)
        or CREDENTIAL_PREFIX.match(value)
        or "://" in value
    ):
        return None
    return value


def _safe_request_id(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not SAFE_REQUEST_ID.fullmatch(value)
        or CREDENTIAL_PREFIX.match(value)
        or "://" in value
    ):
        return None
    return value


def _safe_model(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not SAFE_MODEL.fullmatch(value)
        or CREDENTIAL_PREFIX.match(value)
        or "://" in value
    ):
        return None
    return value


def safe_error_code(value: Any) -> str | None:
    """Expose the fixed grammar for safe outer-runtime classifications."""

    return _safe_code(value)


def provider_request_id_from_headers(headers: Any) -> str | None:
    """Return one validated ID from an explicit allowlist of header names."""

    if not isinstance(headers, Mapping) and not hasattr(headers, "items"):
        return None
    try:
        items = headers.items()
    except (AttributeError, TypeError):
        return None
    lower = {name.lower(): value for name, value in items if isinstance(name, str)}
    for name in REQUEST_ID_HEADERS:
        value = _safe_request_id(lower.get(name))
        if value is not None:
            return value
    return None


def normalize_event(value: Any) -> dict[str, Any] | None:
    """Validate the only event form allowed into native checkpoint storage."""

    if not isinstance(value, Mapping):
        return None
    sequence = value.get("sequence")
    kind = value.get("kind")
    status = value.get("status")
    if (
        type(sequence) is not int
        or not 0 <= sequence <= 2_147_483_647
        or not isinstance(kind, str)
        or kind not in EVENT_KINDS
        or not isinstance(status, str)
        or status not in EVENT_STATUSES
    ):
        return None
    normalized: dict[str, Any] = {"sequence": sequence, "kind": kind, "status": status}
    stage = value.get("stage")
    if isinstance(stage, str) and stage in STAGES:
        normalized["stage"] = stage
    model = _safe_model(value.get("model"))
    if model is not None:
        normalized["model"] = model
    duration_ms = value.get("duration_ms")
    if type(duration_ms) is int and 0 <= duration_ms <= 86_400_000:
        normalized["duration_ms"] = duration_ms
    error_code = _safe_code(value.get("error_code"))
    if error_code is not None:
        normalized["error_code"] = error_code
    http_status = value.get("http_status")
    if type(http_status) is int and 100 <= http_status <= 599:
        normalized["http_status"] = http_status
    request_id = _safe_request_id(value.get("provider_request_id"))
    if request_id is not None:
        normalized["provider_request_id"] = request_id
    if type(value.get("retryable")) is bool:
        normalized["retryable"] = value["retryable"]
    terminal_state = value.get("terminal_state")
    if isinstance(terminal_state, str) and terminal_state in TERMINAL_STATES:
        normalized["terminal_state"] = terminal_state
    mcp_tool = value.get("mcp_tool")
    if isinstance(mcp_tool, str) and mcp_tool in MCP_TOOLS:
        normalized["mcp_tool"] = mcp_tool
    operation = value.get("operation")
    if isinstance(operation, str) and operation in OPERATIONS:
        normalized["operation"] = operation
    count = value.get("count")
    if type(count) is int and 0 <= count <= 1_000_000:
        normalized["count"] = count
    return normalized
