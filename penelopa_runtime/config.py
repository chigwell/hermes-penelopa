"""Read launch credentials once; never expose external credentials to Hermes."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("An HTTP(S) endpoint is required")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoints must not contain credentials, queries, or fragments")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    task_id: str
    user_id: str
    claim_version: int
    api_base: str
    provider_base: str
    model: str
    fallback_models: tuple[str, ...]
    task_token: str
    lifecycle_token: str
    provider_token: str
    home: Path
    internal_timeout: float = 30
    provider_timeout: float = 180
    heartbeat_interval: float = 15

    @classmethod
    def from_env(cls) -> Settings:
        def required(name: str) -> str:
            value = os.environ.get(name, "").strip()
            if not value:
                raise ValueError(f"Missing launch setting: {name}")
            return value

        task_id = str(UUID(required("HERMES_TASK_ID")))
        user_id = str(UUID(required("HERMES_USER_ID")))
        fallbacks = json.loads(os.environ.get("HERMES_FALLBACK_MODELS", "[]"))
        if (
            not isinstance(fallbacks, list)
            or len(fallbacks) > 2
            or not all(isinstance(x, str) and x.strip() for x in fallbacks)
        ):
            raise ValueError("HERMES_FALLBACK_MODELS must contain at most two model names")
        home = Path(os.environ.get("HERMES_HOME", "/opt/data/hermes"))
        if not home.is_absolute():
            raise ValueError("HERMES_HOME must be absolute")
        settings = cls(
            task_id=task_id,
            user_id=user_id,
            claim_version=int(required("HERMES_CLAIM_VERSION")),
            api_base=endpoint(required("HERMES_INTERNAL_API_BASE_URL")),
            provider_base=endpoint(required("HERMES_BASE_URL")),
            model=required("HERMES_MODEL"),
            fallback_models=tuple(fallbacks),
            task_token=required("HERMES_TASK_TOKEN"),
            lifecycle_token=required("HERMES_LIFECYCLE_TOKEN"),
            provider_token=required("HERMES_API_TOKEN"),
            home=home,
            internal_timeout=float(os.environ.get("HERMES_INTERNAL_HTTP_TIMEOUT_SECONDS", "30")),
            provider_timeout=float(os.environ.get("HERMES_PROVIDER_HTTP_TIMEOUT_SECONDS", "180")),
            heartbeat_interval=float(os.environ.get("HERMES_HEARTBEAT_INTERVAL_SECONDS", "15")),
        )
        if settings.claim_version < 1 or any(
            not math.isfinite(value) or value <= 0
            for value in (
                settings.internal_timeout,
                settings.provider_timeout,
                settings.heartbeat_interval,
            )
        ):
            raise ValueError("Claim version and per-request timeouts must be positive")
        # Neither stdio children nor native persistence may inherit external secrets.
        for name in list(os.environ):
            upper = name.upper()
            if any(part in upper for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")):
                os.environ.pop(name, None)
        for name in ("HERMES_BASE_URL", "HERMES_INTERNAL_API_BASE_URL", "HERMES_FALLBACK_MODELS"):
            os.environ.pop(name, None)
        return settings


def bootstrap(home: Path) -> None:
    """A root-only, exact-path migration for old root-owned per-user volumes."""
    if home.is_symlink() or home.parent.is_symlink():
        raise ValueError("Managed runtime directories must not be symlinks")
    workspace = home.parent / "workspace"
    if workspace.is_symlink():
        raise ValueError("Managed workspace must not be a symlink")
    home.parent.mkdir(parents=True, exist_ok=True)
    home.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    if os.geteuid() == 0:
        for path in (
            home.parent,
            home,
            workspace,
            home.parent / "runtime.json",
            home.parent / "pending_terminal.json",
        ):
            if path.is_symlink():
                raise ValueError("Managed runtime paths must not be symlinks")
            if path.exists():
                os.chown(path, 10001, 10001)
        os.setgroups([])
        os.setgid(10001)
        os.setuid(10001)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_WRITE_SAFE_ROOT"] = str(home)
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    for variable, directory in {
        "HOME": "home",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "XDG_CACHE_HOME": "cache",
    }.items():
        path = home / directory
        if path.is_symlink():
            raise ValueError("Managed runtime directories must not be symlinks")
        path.mkdir(exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PATH"] = "/opt/hermes/.venv/bin:/opt/hermes/bin:/usr/local/bin:/usr/bin:/bin"


def write_managed_config(settings: Settings, broker: str) -> None:
    # Import after credentials have been removed and the home has been bound.
    import sys

    import yaml

    auxiliary = {}
    for task, stage in {
        "compression": "compaction",
        "goal_judge": "goal_check",
        "session_search": "session_search",
        "title_generation": "analysis",
    }.items():
        auxiliary[task] = {
            "provider": "custom",
            "model": settings.model,
            "base_url": f"{broker}/provider/{stage}/v1",
            "api_key": "penelopa-local",
        }
    auxiliary["background_review"] = {
        "enabled": True,
        "provider": "auto",
        "model": "",
        "extra_tools": [],
    }
    config = {
        "model": {
            "default": settings.model,
            "provider": "custom",
            "base_url": f"{broker}/provider/analysis/v1",
        },
        "agent": {"max_turns": None, "run_budget_seconds": None, "environment_probe": False},
        "goals": {"max_turns": sys.maxsize},
        "memory": {"memory_enabled": True, "user_profile_enabled": True, "write_approval": False},
        "skills": {
            "inline_shell": False,
            "external_dirs": [],
            "project_discovery": False,
            "trusted_project_dirs": [],
            "guard_agent_created": True,
            "write_approval": False,
            "ledger": True,
        },
        "sessions": {"auto_prune": False, "auto_archive": False},
        "curator": {"enabled": False},
        "checkpoints": {"enabled": False, "auto_prune": False},
        "plugins": {},
        "auxiliary": auxiliary,
        "mcp": {"auto_reload_on_config_change": False},
        "tools": {"tool_search": {"enabled": "off"}},
        "mcp_servers": {
            "penelopa": {
                "command": sys.executable,
                "args": ["-m", "penelopa_runtime.bridge", "--broker", broker],
                "timeout": settings.internal_timeout,
                "connect_timeout": settings.internal_timeout,
            }
        },
    }
    path = settings.home / "config.yaml"
    if path.is_symlink():
        raise ValueError("Managed config must not be a symlink")
    existing = yaml.safe_load(path.read_text()) if path.exists() else {}
    if existing is not None and not isinstance(existing, dict):
        raise ValueError("Invalid managed config")
    def merge(old, managed):
        result = dict(old)
        for key, value in managed.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        return result
    config = merge(existing or {}, config)
    # Only the managed MCP service and no plugins may survive old config.
    config["plugins"] = {}
    config["memory"].pop("provider", None)
    config["mcp_servers"] = {"penelopa": config["mcp_servers"]["penelopa"]}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    os.chmod(path, 0o600)
