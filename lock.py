"""Cross-process lock guarding the workspace during a task run.

A file on disk, not a Python-level lock (threading.Lock, etc.) -
the workspace can be operated on by wholly separate processes with no
shared memory, e.g. a local `python main.py` and a Docker container
both bind-mounted onto the same host workspace/ directory. Only a
file on the filesystem both can see actually closes that race.
Confirmed this was a real gap, not theoretical: a local run and a
Docker run against the same workspace/ raced mid-task and one run's
commit was silently discarded from git history.
"""

from __future__ import annotations

import contextlib
import os
import time

from sandbox import get_workspace_root

LOCK_NAME = ".agent.lock"

# Long enough that no legitimate run - including a decomposed
# multi-subtask task, each with its own repair/coverage/review loops
# - should plausibly still be alive; short enough that a lock left
# behind by a crashed run doesn't block every future run indefinitely.
STALE_AFTER_SECONDS = 6 * 60 * 60


class WorkspaceLockedError(RuntimeError):
    """Raised when another run already holds the workspace lock."""


def _lock_path():
    return get_workspace_root() / LOCK_NAME


def _is_stale(path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > STALE_AFTER_SECONDS


@contextlib.contextmanager
def held():
    """Hold the workspace lock for the duration of the with-block.

    Raises WorkspaceLockedError immediately if another run already
    holds it (and it isn't stale), rather than waiting or queueing -
    a task started while one is already running should fail fast and
    say so, the same contract the old web UI enforced with its 409
    response, just now covering local/Docker runs too.
    """
    path = _lock_path()

    if path.exists() and _is_stale(path):
        path.unlink(missing_ok=True)

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise WorkspaceLockedError(
            "Another task is already running against this workspace "
            f"(lock held by {path}). Wait for it to finish, or if "
            "you're sure no run is actually active (e.g. it crashed "
            "without cleaning up), delete that file and try again."
        ) from None

    with os.fdopen(fd, "w") as f:
        f.write(f"pid={os.getpid()}\n")

    try:
        yield
    finally:
        path.unlink(missing_ok=True)
