"""Pinned Hermes embedding, with a fail-closed model capability boundary."""

from __future__ import annotations

import importlib
import json
import threading
from copy import deepcopy

from penelopa_runtime import MCP_TOOLS, NATIVE_TOOLS

ALLOWED_NAMES = NATIVE_TOOLS | frozenset(f"mcp__penelopa__{name}" for name in MCP_TOOLS)
_installed = False


def permitted(name, args=None):
    if name not in ALLOWED_NAMES:
        return False
    if name == "session_search" and isinstance(args, dict):
        if args.get("profile") not in (None, ""):
            return False
        if "/" in str(args.get("session_id", "")):
            return False
    return True


def install_policy(task_kind="recommendation_generation"):
    """Install BEFORE importing run_agent/model_tools or any tool discovery.

    The adapter owns exposure/dispatch only. It never replaces Hermes's model
    loop, native handlers, memory store, skill store, goal judge or compactor.
    This revision-pinned seam intentionally fails closed on future API drift.
    """
    global _installed, ALLOWED_NAMES
    if task_kind == "self_improvement":
        ALLOWED_NAMES = NATIVE_TOOLS | {"mcp__penelopa__get_task_brief", "mcp__penelopa__complete_self_improvement"}
    if _installed:
        return
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import tools.registry as registry_module

    plugins.discover_plugins = lambda *args, **kwargs: None
    plugins.start_background_plugin_discovery = lambda *args, **kwargs: None
    plugins.invoke_hook = lambda *args, **kwargs: []
    plugins.invoke_middleware = lambda *args, **kwargs: []
    plugins.has_hook = lambda *args, **kwargs: False
    lifecycle.invoke_hook = lambda *args, **kwargs: []
    lifecycle.has_hook = lambda *args, **kwargs: False
    registry = registry_module.registry
    original_register = registry.register
    original_dispatch = registry.dispatch
    original_definitions = registry.get_definitions

    def register(name, *args, **kwargs):
        if name in ALLOWED_NAMES:
            return original_register(name, *args, **kwargs)
        return None

    def dispatch(name, args, **kwargs):
        if not permitted(name, args):
            return json.dumps({"error": "Tool denied by the managed runtime capability policy"})
        return original_dispatch(name, args, **kwargs)

    def definitions(tool_names, *args, **kwargs):
        return original_definitions(set(tool_names) & ALLOWED_NAMES, *args, **kwargs)

    registry.register = register
    registry.dispatch = dispatch
    registry.get_definitions = definitions

    def discover(*args, **kwargs):
        modules = [
            "tools.memory_tool",
            "tools.session_search_tool",
            "tools.skills_tool",
            "tools.skill_manager_tool",
        ]
        for name in modules:
            importlib.import_module(name)
        return modules

    registry_module.discover_builtin_tools = discover
    from tools import session_search_tool

    def deny_profile(*args, **kwargs):
        raise ValueError("Cross-profile session access is unavailable")

    # Native search otherwise scans every profile when an ID misses the active
    # DB, even with no explicit profile argument. Keep its actual DB/search
    # implementation, but remove both cross-profile resolution seams.
    session_search_tool._resolve_profile_db = deny_profile
    session_search_tool._locate_session_db = lambda *args, **kwargs: (None, None)
    import model_tools

    original_handle = model_tools.handle_function_call

    def handle(name, args, *positional, **kwargs):
        if not permitted(name, args):
            return json.dumps({"error": "Tool denied by the managed runtime capability policy"})
        return original_handle(name, args, *positional, **kwargs)

    model_tools.handle_function_call = handle
    _installed = True


def assert_surface(agent, task_kind="recommendation_generation"):
    names = {tool["function"]["name"] for tool in agent.tools}
    required = NATIVE_TOOLS | {
        "mcp__penelopa__get_task_brief",
        "mcp__penelopa__complete_self_improvement" if task_kind == "self_improvement" else "mcp__penelopa__submit_recommendations",
    }
    if not required <= names or not names <= ALLOWED_NAMES:
        raise RuntimeError(
            "native_tool_contract_mismatch: missing="
            + ",".join(sorted(required - names))
            + "; extra="
            + ",".join(sorted(names - ALLOWED_NAMES))
        )


def run_terminal_review(agent, messages, state):
    """Run the genuine upstream review worker once, retaining its whole lifetime."""
    from agent.background_review import (
        finish_background_review_run,
        prepare_background_review_run,
        spawn_background_review_thread,
    )
    from tools.thread_context import propagate_context_to_thread

    if state.stop.is_set():
        state.checkpoint(review_status="interrupted")
        return

    run = prepare_background_review_run(agent)
    if run is None:
        raise RuntimeError("review_already_running")
    target, _ = spawn_background_review_thread(
        agent,
        messages_snapshot=deepcopy(messages),
        review_memory=True,
        review_skills=True,
        focus=None,
        review_run=run,
    )
    state.checkpoint("reviewing", review_status="running")
    original_failure = agent._emit_auxiliary_failure

    def review_failure(*args, **kwargs):
        state.checkpoint(review_status="failed", error_code="native_review_failed")

    agent._emit_auxiliary_failure = review_failure
    worker = threading.Thread(
        target=propagate_context_to_thread(target), name="bg-review", daemon=False
    )
    try:
        worker.start()
    except BaseException:
        finish_background_review_run(agent, run)
        raise
    cancel_requested = False
    while worker.is_alive():
        worker.join(timeout=0.25)
        if state.stop.is_set() and not cancel_requested:
            cancel_requested = True
            from agent.background_review import cancel_background_review_for_live_turn

            cancel_background_review_for_live_turn(agent)
            # Keep ownership until the actual worker exits; no close-vs-write race.
    agent._emit_auxiliary_failure = original_failure
    if state.review_status != "failed":
        state.checkpoint(review_status="interrupted" if state.stop.is_set() else "completed")
