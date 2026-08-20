"""Workspace / security boundary.

The sandbox is the only module allowed to decide what counts as inside
the agent's workspace. Every tool that touches the filesystem must
resolve paths through here first. sandbox.py has no dependencies on
tools.py or agent.py.
"""

from __future__ import annotations

import os
from pathlib import Path


class SandboxViolationError(Exception):
    """Raised when a path would resolve outside the workspace."""


def get_workspace_root() -> Path:
    """Return the workspace root, creating it if needed.

    Honors WORKSPACE_DIR (used to point at /workspace inside the
    Docker container); falls back to the local workspace/ directory
    next to this file.
    """
    override = os.environ.get("WORKSPACE_DIR")
    root = Path(override) if override else Path(__file__).resolve().parent / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def resolve_path(relative_path: str) -> Path:
    """Resolve relative_path against the workspace root.

    Rejects absolute paths and any traversal (".." or a symlink) that
    would land outside the workspace. Does not require the target to
    already exist.
    """
    root = get_workspace_root()

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise SandboxViolationError(f"Absolute paths are not allowed: {relative_path!r}")

    resolved = (root / candidate).resolve()

    if resolved != root and root not in resolved.parents:
        raise SandboxViolationError(f"Path escapes the workspace: {relative_path!r} -> {resolved}")

    return resolved


def resolve_existing_file(relative_path: str) -> Path:
    """Resolve relative_path and require it to be an existing file."""
    path = resolve_path(relative_path)
    if not path.is_file():
        raise SandboxViolationError(f"Not a file in the workspace: {relative_path!r}")
    return path


def resolve_existing_dir(relative_path: str) -> Path:
    """Resolve relative_path and require it to be an existing directory."""
    path = resolve_path(relative_path)
    if not path.is_dir():
        raise SandboxViolationError(f"Not a directory in the workspace: {relative_path!r}")
    return path
