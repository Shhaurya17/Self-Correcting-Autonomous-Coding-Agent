"""Git-based checkpointing for the workspace.

Not exposed to the agent as a tool - agent.py's orchestration calls
these directly around each attempt, so checkpoint/rollback decisions
are deterministic, not left to the LLM to decide when to invoke.

Operates on a git repo rooted at the workspace itself (nested inside
this project's own repo). ensure_repo() creates one if the workspace
doesn't already have one - if the agent is pointed at an existing
project that's already a git repo, it's used as-is.
"""

from __future__ import annotations

import subprocess

from sandbox import get_workspace_root


def _run(*args: str) -> subprocess.CompletedProcess:
    root = get_workspace_root()
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


_DEFAULT_GITIGNORE = """\
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
.agent.lock
"""


def ensure_repo() -> None:
    """Initialize a git repo in the workspace if one doesn't exist yet.

    Only seeds a .gitignore when it's the one creating the repo - an
    existing repo (e.g. a real project the agent was pointed at) is
    used as-is, not second-guessed.
    """
    root = get_workspace_root()
    if (root / ".git").exists():
        return
    _run("init", "-q")
    _run("config", "user.email", "agent@coding-agent.local")
    _run("config", "user.name", "Coding Agent")

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")


def checkpoint(message: str = "checkpoint") -> str | None:
    """Commit the current workspace state and return the commit hash.

    If the tree is already clean (nothing to commit), returns the
    existing HEAD hash instead of creating an empty commit - or None
    if there is no commit yet at all.
    """
    ensure_repo()
    _run("add", "-A")

    status = _run("status", "--porcelain")
    if not status.stdout.strip():
        rev = _run("rev-parse", "HEAD")
        return rev.stdout.strip() if rev.returncode == 0 else None

    _run("commit", "-q", "-m", message)
    rev = _run("rev-parse", "HEAD")
    return rev.stdout.strip() if rev.returncode == 0 else None


def commit(message: str) -> str | None:
    """Ensure HEAD carries message - for a final, deliberate commit.

    Unlike checkpoint(), this doesn't silently keep an earlier
    checkpoint's generic message when the tree happens to already be
    clean at this point (the common case: nothing changed between the
    last safety checkpoint and here). If there's nothing new to
    commit, the existing HEAD is relabeled via amend instead of left
    with whatever message the last checkpoint used.
    """
    ensure_repo()
    _run("add", "-A")

    status = _run("status", "--porcelain")
    if status.stdout.strip():
        _run("commit", "-q", "-m", message)
    else:
        head = _run("rev-parse", "HEAD")
        if head.returncode == 0:
            _run("commit", "--amend", "-q", "-m", message)

    rev = _run("rev-parse", "HEAD")
    return rev.stdout.strip() if rev.returncode == 0 else None


def rollback(to: str) -> str:
    """Hard reset the workspace to a previous checkpoint commit."""
    ensure_repo()
    result = _run("reset", "--hard", to)
    return (result.stdout + result.stderr).strip()


def get_diff(since: str | None = None, stat_only: bool = False) -> str:
    """Return the diff of uncommitted changes, or since a given commit."""
    ensure_repo()
    args = ["diff"]
    if stat_only:
        args.append("--stat")
    if since:
        args.append(since)
    result = _run(*args)
    return result.stdout
