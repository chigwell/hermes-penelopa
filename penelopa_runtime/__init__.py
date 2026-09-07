"""Deployment/lifecycle adapter; the reasoning and tool loop belong to Hermes."""

UPSTREAM_REVISION = "29112bef099274229cadff79cdff7bf7b99c4b77"
NATIVE_TOOLS = frozenset({"memory", "skills_list", "skill_view", "skill_manage", "session_search"})
MCP_TOOLS = frozenset(
    {
        "get_task_brief",
        "list_user_sessions",
        "list_assigned_sessions",
        "collect_goal_evidence",
        "read_session_events",
        "get_event_window",
        "get_session_process_summary",
        "compare_snapshot_sessions",
        "get_step_evidence",
        "remember_observation",
        "validate_recommendations",
        "submit_recommendations",
    }
)
