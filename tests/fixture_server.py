"""Synthetic MCP/lifecycle/OpenAI fixture. Never uses a real key or user transcript."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PUBLIC_TOOLS = (
    "get_task_brief", "list_assigned_sessions", "list_user_sessions",
    "collect_goal_evidence", "read_session_events", "get_event_window",
    "get_session_process_summary", "compare_snapshot_sessions", "get_step_evidence",
    "remember_observation", "validate_recommendations", "submit_recommendations",
)


@dataclass
class FixtureCase:
    user_id: str
    label: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_token: str = field(default_factory=lambda: "synthetic-task-" + uuid.uuid4().hex)
    lifecycle_token: str = field(default_factory=lambda: "synthetic-lifecycle-" + uuid.uuid4().hex)
    provider_token: str = field(default_factory=lambda: "synthetic-provider-" + uuid.uuid4().hex)
    accepted: threading.Event = field(default_factory=threading.Event)
    review_started: threading.Event = field(default_factory=threading.Event)
    release_review: threading.Event = field(default_factory=threading.Event)
    main_step: int = 0
    review_step: int = 0
    terminal_calls: int = 0
    lost_ack: bool = False
    force_fallback: bool = False
    prose_before_tools: bool = False
    provider_requests: list[dict] = field(default_factory=list)
    mcp_calls: list[str] = field(default_factory=list)
    heartbeats: list[dict] = field(default_factory=list)
    terminal_meta: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def sentinel(self):
        return f"Synthetic {self.label} prefers reproducible tests before releases."

    @property
    def review_sentinel(self):
        return f"Synthetic {self.label} values persisted task-boundary review."

    def env(self, url: str) -> dict[str, str]:
        return {
            "HERMES_USER_ID": self.user_id,
            "HERMES_TASK_ID": self.task_id,
            "HERMES_CLAIM_VERSION": "1",
            "HERMES_TASK_TOKEN": self.task_token,
            "HERMES_LIFECYCLE_TOKEN": self.lifecycle_token,
            "HERMES_INTERNAL_API_BASE_URL": url,
            "HERMES_BASE_URL": url + "/v1",
            "HERMES_MODEL": "fixture-primary",
            "HERMES_FALLBACK_MODELS": '["fixture-fallback"]',
            "HERMES_API_TOKEN": self.provider_token,
            "HERMES_PROVIDER_TYPE": "custom",
            "HERMES_PROVIDER_HTTP_TIMEOUT_SECONDS": "40",
            "HERMES_INTERNAL_HTTP_TIMEOUT_SECONDS": "10",
        }


def tool_schema(name: str) -> dict:
    properties = {}
    required = []
    if name == "read_session_events":
        properties = {"session_id": {"type": "string"}, "limit": {"type": "integer"}}
        required = ["session_id"]
    elif name == "submit_recommendations":
        properties = {
            "results": {"type": "array", "items": {"type": "object"}},
            "outcome": {"type": "string", "enum": ["results", "no_recommendation"]},
            "reason": {"type": "string"},
        }
        required = ["results"]
    return {"name": name, "description": "Synthetic fixture " + name,
            "inputSchema": {"type": "object", "properties": properties, "required": required}}


class FixtureBackend:
    def __init__(self):
        self.cases: list[FixtureCase] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def respond(self, value, code=200):
                body = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def case_for(self, kind):
                bearer = self.headers.get("Authorization", "").removeprefix("Bearer ")
                case = next((c for c in fixture.cases if getattr(c, kind + "_token") == bearer), None)
                if case is None:
                    self.respond({"detail": "Unknown synthetic credential"}, 401)
                return case

            def do_GET(self):
                if self.path.endswith("/receipt"):
                    case = self.case_for("lifecycle")
                    if case:
                        self.respond({"accepted": case.accepted.is_set(), "task_id": case.task_id,
                                      "claim_version": 1,
                                      "status": "SUCCEEDED" if case.accepted.is_set() else "RUNNING",
                                      "receipt": {"task_id": case.task_id, "status": "SUCCEEDED"}
                                      if case.accepted.is_set() else None})
                else:
                    self.respond({"detail": "No SSE stream"}, 405)

            def do_DELETE(self):
                self.respond({"detail": "Task token revoked"}, 401)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                if self.path == "/v1/chat/completions":
                    case = self.case_for("provider")
                    if case:
                        self.provider(case, body)
                elif self.path.endswith("/runtime-heartbeat"):
                    case = self.case_for("lifecycle")
                    if case:
                        case.heartbeats.append(body)
                        self.respond({"ok": True, "task_id": case.task_id, "claim_version": 1,
                                      "status": "SUCCEEDED" if case.accepted.is_set() else "RUNNING"})
                elif self.path.rstrip("/") == "/_internal/hermes/mcp":
                    case = self.case_for("task")
                    if case:
                        self.mcp(case, body)
                else:
                    self.respond({"detail": "Unknown fixture route"}, 404)

            def mcp(self, case, body):
                if case.accepted.is_set():
                    self.respond({"detail": "Task token revoked"}, 401)
                    return
                method = body.get("method")
                params = body.get("params", {})
                if method == "initialize":
                    result = {"protocolVersion": params.get("protocolVersion", "2025-03-26"),
                              "capabilities": {"tools": {}},
                              "serverInfo": {"name": "synthetic-penelopa", "version": "1"}}
                elif method in {"notifications/initialized", "notifications/cancelled"}:
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                elif method == "tools/list":
                    result = {"tools": [tool_schema(name) for name in PUBLIC_TOOLS] + [
                        {**tool_schema(name), "_meta": {"hermes/internal": True}}
                        for name in ("heartbeat_task", "report_failure", "record_provider_model_usage")
                    ]}
                elif method == "tools/call":
                    name = params.get("name")
                    args = params.get("arguments", {})
                    case.mcp_calls.append(name)
                    if name == "get_task_brief":
                        data = {"task_id": case.task_id, "runtime_engine": "nous_hermes",
                                "execution_policy": "goal_driven", "scope_mode": "user_history",
                                "analysis_goal": {"objective": "Inspect synthetic retained user work. "
                                                  "Submit a grounded recommendation or no_recommendation with reason."},
                                "allowed_sessions": [{"id": case.session_id}], "learning_context": {}}
                    elif name in {"list_user_sessions", "list_assigned_sessions"}:
                        data = {"sessions": [{"id": case.session_id}], "next_cursor": None}
                    elif name == "read_session_events":
                        if args.get("session_id") != case.session_id:
                            data = {"error": "Session does not belong to token owner"}
                            result = {"content": [{"type": "text", "text": json.dumps(data)}], "isError": True}
                            self.respond({"jsonrpc": "2.0", "id": body.get("id"), "result": result})
                            return
                        data = {"events": [{"event_id": case.event_id, "session_id": case.session_id,
                                            "content_text": case.sentinel}], "next_cursor": None}
                    elif name == "submit_recommendations":
                        case.terminal_calls += 1
                        if args.get("results") != [] or args.get("outcome") != "no_recommendation" or not args.get("reason"):
                            case.errors.append("Invalid terminal contract")
                            result = {"isError": True, "content": [{"type": "text", "text": "Invalid terminal payload"}]}
                            self.respond({"jsonrpc": "2.0", "id": body.get("id"), "result": result})
                            return
                        case.terminal_meta = params.get("_meta", {})
                        case.accepted.set()
                        if case.lost_ack:
                            self.close_connection = True
                            self.connection.shutdown(2)
                            return
                        data = {"task_id": case.task_id, "status": "SUCCEEDED"}
                    elif name == "report_failure":
                        case.errors.append("Runtime reported failure: " + json.dumps(args))
                        data = {"status": "FAILED"}
                    elif name == "heartbeat_task":
                        data = {"ok": True, "task_id": case.task_id, "status": "RUNNING",
                                "lease_expires_at": (datetime.now(timezone.utc) + timedelta(seconds=180)).isoformat()}
                    else:
                        data = {"ok": True, "task_id": case.task_id, "status": "RUNNING"}
                    result = {"content": [{"type": "text", "text": json.dumps(data)}], "isError": False}
                    if name != "submit_recommendations":
                        result["structuredContent"] = data
                else:
                    result = {}
                self.respond({"jsonrpc": "2.0", "id": body.get("id"), "result": result})

            def provider(self, case, body):
                stage = self.headers.get("X-Penelopa-Stage", "analysis")
                case.provider_requests.append({**body, "_fixture_stage": stage})
                if case.force_fallback and body.get("model") == "fixture-primary":
                    self.respond({"error": {"message": "Synthetic availability failure"}}, 503)
                    return
                tools = {item.get("function", {}).get("name"): item.get("function", {})
                         for item in body.get("tools", [])}
                call = None
                content = "Done."
                if stage == "background_review" and "memory" in tools:
                    if case.review_step == 0:
                        case.review_started.set()
                        if not case.release_review.wait(25):
                            case.errors.append("Test did not release review")
                        call = ("memory", {"target": "memory", "action": "add", "content": case.review_sentinel})
                    case.review_step += 1
                elif any(name and name.endswith("get_task_brief") for name in tools):
                    def mcp_name(suffix):
                        return next(name for name in tools if name.endswith(suffix))
                    steps = [
                        (mcp_name("get_task_brief"), {}),
                        (mcp_name("list_user_sessions"), {}),
                        (mcp_name("read_session_events"), {"session_id": case.session_id}),
                        ("memory", {"target": "user", "action": "add", "content": case.sentinel}),
                        ("skill_manage", {"operations": [{"name": "fixture-" + case.label, "action": "create",
                            "content": "---\nname: fixture-" + case.label + "\ndescription: Use when testing synthetic persistence.\n---\n# Synthetic skill\nVerify evidence before release.\n"}]}),
                        (mcp_name("submit_recommendations"), {"results": [], "outcome": "no_recommendation",
                            "reason": "Synthetic retained context has no additional useful intervention."}),
                    ]
                    if case.prose_before_tools and case.main_step == 0:
                        content = "I am done."
                    else:
                        step = case.main_step - int(case.prose_before_tools)
                        if step < len(steps):
                            call = steps[step]
                    case.main_step += 1
                    if case.main_step > 12:
                        case.errors.append("Native goal loop failed to reach terminal")
                        self.respond({"error": {"message": "Fixture loop guard"}}, 400)
                        return
                else:
                    # The main business receipt, not this model verdict, is authoritative.
                    content = "CONTINUE"
                message = {"role": "assistant", "content": content if call is None else None}
                if call:
                    message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex,
                        "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
                response = {"id": "chatcmpl-synthetic", "object": "chat.completion", "created": int(time.time()),
                            "model": body.get("model"), "choices": [{"index": 0, "message": message,
                            "finish_reason": "tool_calls" if call else "stop"}],
                            "usage": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}}
                if not body.get("stream"):
                    self.respond(response)
                    return
                delta = {**message}
                for index, tool_call in enumerate(delta.get("tool_calls", [])):
                    tool_call["index"] = index
                chunks = [
                    {**response, "object": "chat.completion.chunk", "usage": None,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {**response, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if call else "stop"}]},
                ]
                payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                encoded = payload.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def container_url(self):
        return f"http://host.docker.internal:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        for case in self.cases:
            case.release_review.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
