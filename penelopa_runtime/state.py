"""Non-secret atomic checkpoints and uncapped, aggregated usage accounting."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from penelopa_runtime.diagnostics import MAX_EVENTS, OPERATIONS, VERSION, normalize_event


def atomic_json(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError("Managed checkpoint must not be a symlink")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".penelopa-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class RuntimeState:
    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.phase = "starting"
        self.session_id = None
        self.review_status = "pending"
        self.accepted = None
        self.stop = threading.Event()
        self.metrics = {}
        self.error_code = None
        self.requests = 0
        self.missing_usage_calls = 0
        self.last_model = settings.model
        self.diagnostic_events = []
        self.diagnostic_sequence = 0
        self.operation_counts = {}
        self.path = settings.home.parent / "runtime.json"
        self.restore_attempt_metrics()

    def restore_attempt_metrics(self):
        if self.path.is_symlink():
            raise ValueError("Managed checkpoint must not be a symlink")
        if not self.path.exists():
            return
        previous = json.loads(self.path.read_text())
        if not isinstance(previous, dict):
            raise ValueError("Invalid runtime checkpoint")
        if any(
            previous.get(key) != getattr(self.settings, key)
            for key in ("task_id", "user_id", "claim_version")
        ):
            return
        models = {self.settings.model, *self.settings.fallback_models}
        stages = {"analysis", "goal_check", "compaction", "session_search", "background_review"}
        metrics = previous.get("generation_stages", [])
        if not isinstance(metrics, list) or len(metrics) > 15:
            raise ValueError("Invalid runtime metric checkpoint")
        for metric in metrics:
            if (
                not isinstance(metric, dict)
                or not isinstance(metric.get("stage"), str)
                or metric["stage"] not in stages
                or not isinstance(metric.get("model"), str)
                or metric["model"] not in models
                or type(metric.get("duration_seconds")) not in (int, float)
                or not math.isfinite(metric["duration_seconds"])
                or metric["duration_seconds"] < 0
            ):
                raise ValueError("Invalid runtime metric checkpoint")
            self.metrics[(metric["stage"], metric["model"])] = metric
        for key in ("llm_requests", "missing_usage_calls"):
            value = previous.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError("Invalid runtime request checkpoint")
            setattr(self, "requests" if key == "llm_requests" else key, value)
        if isinstance(previous.get("provider_model"), str) and previous["provider_model"] in models:
            self.last_model = previous["provider_model"]
        if previous.get("review_status") == "completed":
            self.review_status = "completed"
        events = previous.get("diagnostic_events", [])
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise ValueError("Invalid runtime diagnostic checkpoint")
        sequence = -1
        for item in events:
            event = normalize_event(item)
            if event is None or event["sequence"] <= sequence:
                raise ValueError("Invalid runtime diagnostic checkpoint")
            self.diagnostic_events.append(event)
            sequence = event["sequence"]
        self.diagnostic_sequence = sequence + 1
        operation_counts = previous.get("operation_counts", {})
        if not isinstance(operation_counts, dict) or len(operation_counts) > len(OPERATIONS):
            raise ValueError("Invalid runtime operation checkpoint")
        for operation, count in operation_counts.items():
            if operation not in OPERATIONS or type(count) is not int or not 0 <= count <= 1_000_000:
                raise ValueError("Invalid runtime operation checkpoint")
        self.operation_counts = deepcopy(operation_counts)

    def snapshot(self):
        with self.lock:
            return {
                "task_id": self.settings.task_id,
                "user_id": self.settings.user_id,
                "claim_version": self.settings.claim_version,
                "runtime_engine": "nous_hermes",
                "execution_policy": "goal_driven",
                "runtime_phase": self.phase,
                "heartbeat_at": datetime.now(UTC).isoformat(),
                "session_id": self.session_id,
                "review_status": self.review_status,
                "terminal_accepted": self.accepted is not None,
                "generation_stages": deepcopy(list(self.metrics.values())),
                "llm_requests": self.requests,
                "missing_usage_calls": self.missing_usage_calls,
                "provider_model": self.last_model,
                "error_code": self.error_code,
                "diagnostics_version": VERSION,
                "diagnostic_events": deepcopy(self.diagnostic_events),
                "operation_counts": deepcopy(self.operation_counts),
            }

    def checkpoint(self, phase=None, **fields):
        with self.lock:
            if phase:
                self.phase = phase
            for key, value in fields.items():
                setattr(self, key, value)
            atomic_json(self.path, self.snapshot())

    def record(self, stage, model, duration, usage=None, error=None):
        with self.lock:
            self.requests += 1
            # Five stages times at most three configured models. A long run or
            # many different HTTP errors never overflows the 128-metric wire shape.
            key = (stage, model)
            metric = self.metrics.setdefault(
                key,
                {
                    "stage": stage,
                    "model": model,
                    "duration_seconds": 0,
                    "usage": None,
                    "error_code": error,
                },
            )
            metric["duration_seconds"] = round(metric["duration_seconds"] + duration, 3)
            metric["error_code"] = error
            if error is None and stage == "analysis":
                self.last_model = model
            if error is None and not (
                isinstance(usage, dict)
                and all(
                    isinstance(usage.get(name), int)
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
            ):
                self.missing_usage_calls += 1
            if isinstance(usage, dict):
                target = metric["usage"] or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                for key in target:
                    value = usage.get(key)
                    if isinstance(value, int) and value >= 0:
                        target[key] += value
                metric["usage"] = target
            self.checkpoint()

    def record_diagnostic_event(self, kind, status, **fields):
        """Append one bounded, schema-validated non-content fact and checkpoint it.

        This is intentionally the only way runtime code can add the event.  It
        prevents a provider reply, an MCP argument, or a raw exception from
        being copied wholesale into the persistent user volume.
        """

        with self.lock:
            event = normalize_event(
                {
                    "sequence": self.diagnostic_sequence,
                    "kind": kind,
                    "status": status,
                    **fields,
                }
            )
            if event is None:
                raise ValueError("Invalid native diagnostic event")
            self.diagnostic_sequence += 1
            self.diagnostic_events.append(event)
            if len(self.diagnostic_events) > MAX_EVENTS:
                self.diagnostic_events = self.diagnostic_events[-MAX_EVENTS:]
            operation = event.get("operation")
            count = event.get("count")
            if isinstance(operation, str) and type(count) is int:
                self.operation_counts[operation] = max(self.operation_counts.get(operation, 0), count)
            try:
                atomic_json(self.path, self.snapshot())
            except OSError:
                # Diagnostics must never turn a provider/MCP outcome into a
                # crash merely because an otherwise recoverable volume write
                # failed. Keep the bounded fact in memory for a later regular
                # checkpoint if the volume recovers.
                pass
            return deepcopy(event)

    def next_operation_count(self, operation):
        """Reserve a claim-persistent monotonically increasing operation count."""

        if operation not in OPERATIONS:
            raise ValueError("Invalid native diagnostic operation")
        with self.lock:
            current = self.operation_counts.get(operation, 0)
            if current >= 1_000_000:
                raise ValueError("Native diagnostic operation count exhausted")
            self.operation_counts[operation] = current + 1
            return current + 1

    def terminal_meta(self):
        snapshot = self.snapshot()
        return {
            "io.auto-improve/hermes": {
                "model": self.last_model,
                "stage_metrics": snapshot["generation_stages"],
            }
        }
