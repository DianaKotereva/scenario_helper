"""Internal utilities for config merging and tool helpers."""

import asyncio
import re
from typing import Any, TypeVar

from pydantic import BaseModel

_AGENT_ID_PATTERN = r"\w+_[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}"

_AGENT_ID_RE = re.compile(r"^" + _AGENT_ID_PATTERN + r"$", re.IGNORECASE)
_AGENT_ID_SEARCH_RE = re.compile(r"\b(" + _AGENT_ID_PATTERN + r")\b", re.IGNORECASE)


def is_agent_id(value: str) -> bool:
    """Return True if value matches the agent ID format (name_uuid)."""
    return bool(_AGENT_ID_RE.match(value))


T = TypeVar("T", bound=BaseModel)


def config_from_kwargs(config_class: type[T], base: T | None, kwargs: dict[str, Any]) -> T:
    """Build a config instance from base (optional) and kwargs; kwargs override
    base.

    Generic helper for any Pydantic config model. Used by tools to merge
    agent-level config with per-tool kwargs from the tools array (global or inline).

    Args:
        config_class: Pydantic model class to instantiate (e.g. WebSearchConfig).
        base: Existing config instance, or None to use only kwargs (with model defaults).
        kwargs: Overrides; keys present here override base. None values are skipped.

    Returns:
        New instance of config_class with merged values.
    """
    data = base.model_dump() if base is not None else {}
    data.update({k: v for k, v in kwargs.items() if v is not None})
    return config_class(**data)


async def run_process_with_timeout(
    process: asyncio.subprocess.Process,
    timeout_seconds: float,
) -> tuple[str, str, int]:
    """Run process.communicate() with timeout and decode stdout/stderr.

    Args:
        process: Subprocess created with create_subprocess_exec or create_subprocess_shell.
        timeout_seconds: Max time to wait for process completion.

    Returns:
        Tuple of (stdout_str, stderr_str, return_code).

    Raises:
        asyncio.TimeoutError: If process does not finish within timeout_seconds.
    """
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        process.returncode or 0,
    )
