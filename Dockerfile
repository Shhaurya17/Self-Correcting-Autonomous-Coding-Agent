FROM python:3.13-slim

# Git is required by git_manager.py's checkpointing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-installed for the agent's own run_command/run_tests/check_coverage
# calls inside the container. Avoids the agent needing to pip-install
# these itself at runtime, which would otherwise hit a permission
# error once running as a non-root user below (no write access to the
# system site-packages).
RUN pip install --no-cache-dir pytest pytest-cov

COPY main.py coordinator.py agent.py planner.py reviewer.py tools.py git_manager.py observability.py memory.py lock.py sandbox.py ./

# The agent's operating boundary. Populated at runtime by mounting
# a host directory here (docker run -v <host-dir>:/workspace) —
# never the whole host filesystem.
RUN mkdir -p /workspace

# sandbox.py defaults to a workspace/ directory next to the code
# (/app/workspace here) unless told otherwise. Without this, the
# agent would silently operate on that internal, non-mounted
# directory instead of the bind-mounted /workspace above - a real
# bug caught by actually running a task end to end in the container,
# not something to leave for every `docker run` to remember via -e.
ENV WORKSPACE_DIR=/workspace

# Non-root execution: the agent runs arbitrary shell commands via
# run_command, so it shouldn't be able to do so as root even though
# it's already confined to this disposable container, not the host.
# --create-home so $HOME exists for git's config/safe.directory lookups.
RUN useradd --create-home --uid 1000 agent \
    && chown -R agent:agent /app /workspace
USER agent

# Bind-mounted /workspace is host-owned, not agent-owned - git treats
# that as "dubious ownership" and refuses to operate on it by default.
# This is a real, observed failure (not a hypothetical), confirmed by
# actually running git inside the container against a bind mount.
RUN git config --global --add safe.directory /workspace

# GEMINI_API_KEY is supplied at `docker run` time (-e GEMINI_API_KEY=...)
# and intentionally not declared here — it's never baked into the image.

CMD ["python", "main.py"]
