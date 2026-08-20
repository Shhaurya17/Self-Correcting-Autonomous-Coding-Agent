# Coding Agent

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

An autonomous coding agent built on the OpenAI Agents SDK, configured
against Gemini via Gemini's OpenAI-compatible endpoint. It reads,
writes, edits, and runs code inside a restricted workspace directory,
on a task given via the CLI (`main.py`).

**Status: V4 complete — Multi-Agent Task Decomposition, Persistent Memory.**
Before acting, a read-only planning
pass (`planner.py`) inspects the workspace - including a codebase-wide
text search - and produces a short ordered plan, which the coding agent
uses as a starting point (and can deviate from if inspection shows it's
wrong). The agent can list directories, search the codebase, read/write/edit
files, and run commands (e.g. pytest), sandboxed to `workspace/`. Every
task starts with a git checkpoint of the workspace (`git_manager.py`
initializes a nested repo inside `workspace/` if one doesn't already
exist). After each run, the test suite is re-run independently of the
agent's own claim; on failure it gets fed back the failing output and
another attempt, up to a repair-attempt cap - if it never recovers, the
workspace is rolled back to the pre-task checkpoint rather than left
broken. Once tests pass, a coverage check looks for completely untested
source files and gives the agent more turns to add tests for them,
rolling back to the last known-good checkpoint if a coverage fix breaks
something. Finally, a second agent (`reviewer.py`) reviews the actual
diff since the task's baseline for correctness, security, test quality,
maintainability, unrelated changes, and regressions - unapproved changes
get fed back for more fixes (with the same rollback-on-breakage safety
net), and if still unresolved after that, the result is still returned
since the code works, but with the review's concerns surfaced rather
than hidden. A clean approval ends in a final descriptive commit.
Every run writes a structured audit record (`observability.py`) to
`logs/runs.jsonl` - turns, tool calls, token usage, attempt counts,
test results, and whether the outcome was verified - regardless of
whether the task succeeded, and conversation history handed into each
retry loop is capped rather than left to grow unbounded. Rather than
going silent until a final wall of text, `main.py` shows a live
checklist as the task moves through each stage (analyzing, coding,
testing, fixing, reviewing, finalizing) - `agent.py` only reports
plain stage-name strings via a callback, with no idea whether or how
they're displayed; that rendering, including a plain-ASCII fallback
when the terminal can't render Unicode symbols, lives entirely in
`main.py`. The container this project's Dockerfile builds runs the
agent as a non-root user with a secret-stripped subprocess environment
and capped command output, so `run_command` is contained by more than
convention alone.

Before any of that, a coordinator (`coordinator.py`) decides whether a
task is actually several largely independent pieces of work. Most
tasks aren't - that's a thin pass-through, one extra decision call
then the pipeline above unchanged. For a genuinely multi-part task, it
splits the work into subtasks and runs each one through that *same*
full pipeline in turn, sequentially, against the same workspace/git
history - a later subtask's planning step naturally sees earlier
subtasks' committed work. If a subtask genuinely fails (not just an
unresolved review note), later subtasks aren't attempted rather than
building on broken state.

Each task also carries forward durable project knowledge across
separate runs: after a task finishes, `memory.py` distills anything
worth remembering ("dataclasses are used for all data models",
"chose JWT over session cookies", not raw logs - git history and
`logs/runs.jsonl` already cover that) into `workspace/.agent_memory.json`,
and the next task's planning step is grounded in it. Confirmed this
actually changes behavior, not just plumbing that runs and does
nothing: a task establishing "use dataclasses for all models" was
followed by a second task that never mentioned dataclasses at all,
and it produced one anyway, matching the established convention.

## Architecture

The project is organized as a one-directional dependency graph - nothing
lower in the chain ever imports something higher up:

    main.py -> coordinator.py -> agent.py -> {planner.py, reviewer.py, memory.py, tools.py, git_manager.py}
                              -> lock.py

Where planner.py and reviewer.py both sit on top of tools.py in turn,
and tools.py, git_manager.py, memory.py, and lock.py all sit on top of
sandbox.py directly - nothing lower ever imports something higher up
the chain.

- `main.py` is the entry point calling `coordinator.run_task()` - a
  terminal CLI. It contains no agent reasoning; it just turns
  coordinator.py's plain callback-based progress reports into a live
  checklist.
- `coordinator.py` decides whether a task should be decomposed into
  subtasks, and if so, runs each one through `agent.py` in turn. Not a
  parallel/duplicate agent - it delegates to the exact same pipeline
  agent.py already provides, once per subtask. Holds `lock.py`'s
  workspace lock for the full duration - decomposition decision plus
  every subtask - so a second run can't start partway through.
- `agent.py` is the agent's "brain": model config, instructions, the
  plan-then-code-then-self-correct-then-review orchestration, and git
  checkpoint/rollback around each stage. It does not talk to the
  sandbox directly.
- `planner.py` produces a short ordered plan for a task before any
  writing/editing/running happens, using only the read-only tools
  (`list_directory`, `read_file`, `search_code`).
- `reviewer.py` reviews the diff of what changed (via `git_manager`)
  for correctness, security, test quality, maintainability, unrelated
  changes, and regressions, using the same read-only tools as the
  planner. Returns a structured verdict (approved/summary/issues), not
  free text.
  Neither `planner.py` nor `reviewer.py` import `agent.py`; `agent.py`
  builds the model once and passes it to both, so there's no cycle.
- `tools.py` implements the agent's tools (directory listing, codebase
  text search, file read/write/edit, command execution) plus
  `run_tests` and `check_coverage`, non-agent-facing helpers `agent.py`
  uses to verify results and test coverage independently of what the
  agent claims. Each tool delegates path and command validation to the
  sandbox.
- `git_manager.py` provides `checkpoint`/`commit`/`rollback`/`get_diff`
  over a git repo rooted at the workspace itself. Not exposed to the
  agent as a tool - `agent.py`'s orchestration calls it directly, so
  checkpoint/rollback decisions are deterministic rather than left to
  the LLM to decide when to invoke.
- `observability.py` accumulates a `RunRecord` across a task (turns,
  tool calls, token usage, attempt counts, test results) and writes it
  to `logs/runs.jsonl` once at the end. Not agent-facing, and doesn't
  judge anything - it just records what happened.
- `memory.py` distills durable facts worth remembering from a
  completed task and stores them in `workspace/.agent_memory.json` -
  project knowledge, not this tool's own operational state, so it
  lives inside the workspace rather than alongside `logs/`. Not
  agent-facing; `agent.py` injects it into `planner.py`'s task input,
  planner.py itself has no memory-specific code.
- `lock.py` guards the workspace with a file-based lock
  (`workspace/.agent.lock`) held for a run's full duration - a
  filesystem lock rather than a Python-level one because the workspace
  can be operated on by wholly separate processes with no shared
  memory (a local `python main.py` and a Docker container both
  bind-mounted onto the same host `workspace/`). A second run started
  while one is already active fails fast with a clear error instead of
  racing on the same git history. Auto-expires after 6 hours in case a
  crashed run left it behind.
- `sandbox.py` is the security boundary: workspace path validation, path
  traversal protection, and execution restrictions. It has no
  dependencies on the other project modules.

## Project structure

    coding-agent/
    ├── main.py             CLI entry point
    ├── coordinator.py      Task decomposition & subtask delegation
    ├── agent.py            Agent definition: planning, execution, self-correction, review
    ├── planner.py          Read-only planning pass, run before the agent acts
    ├── reviewer.py         Read-only review pass, run after the agent finishes
    ├── observability.py    Structured audit logging (internal only)
    ├── memory.py           Persistent project memory (internal only)
    ├── tools.py            Agent tools, plus run_tests/check_coverage (internal only)
    ├── git_manager.py      Git checkpoint/rollback (internal only)
    ├── lock.py             Cross-process workspace lock (internal only)
    ├── sandbox.py          Security boundary
    ├── requirements.txt
    ├── Dockerfile
    ├── .env                Local API credentials (not committed)
    ├── .env.example
    ├── .gitignore
    ├── logs/               Audit records (runs.jsonl), not committed
    └── workspace/          Directory the agent operates on

## Installation

Requires Python 3.12+ (developed against 3.13).

    python -m venv .venv
    .venv\Scripts\activate      # Windows
    pip install -r requirements.txt

## Environment setup

Copy `.env.example` to `.env` and fill in your key:

    GEMINI_API_KEY=your-key-here

`.env` is git-ignored and is never read into source, README, Dockerfile,
or logs. The API key is never hard-coded.

Optionally set `GEMINI_MODEL` to override the default model
(`gemini-3.5-flash-lite`, chosen for free-tier quota headroom over the
larger `gemini-3.6-flash`).

## Running locally

    python main.py

Prompts for a task, shows a live progress checklist as it moves
through planning/coding/testing/review, then prints the final result.

## How to use

Everything the agent reads and writes lives under `workspace/` -
describe tasks in terms of files relative to that directory, not
absolute paths (the sandbox rejects those anyway; see
[Security concept](#security-concept)).

    ========================================
     Autonomous Coding Agent
    ========================================

    Task:
    > add a add(a, b) function to math_utils.py with tests

      ✓ analyzing
      ✓ coding
      ✓ testing
      ✓ reviewing
      ⟳ finalizing...

    Task complete. Added `add()` to workspace/math_utils.py with a
    passing test in workspace/test_math_utils.py. Committed as
    a1b2c3d.

A few things worth knowing before writing a task:

- **First run in a fresh `workspace/`**: `git_manager.py` initializes
  a nested git repo there automatically - nothing to set up by hand.
- **One task per run**: `main.py` asks for a single task, runs it to
  completion (or rollback), and exits. Run it again for the next task.
- **Multi-part tasks get split automatically**: phrase a task as
  several distinct pieces of work ("add a `User` model, then a
  `/login` endpoint that uses it") and `coordinator.py` runs each part
  through the full pipeline in sequence, so later parts see earlier
  parts' committed work.
- **Project conventions persist across runs**: once the agent
  establishes a pattern (e.g. "use dataclasses for models"), later
  tasks pick it up automatically via `workspace/.agent_memory.json` -
  no need to repeat it in every task description.
- **Failed tasks don't leave a broken workspace**: if the agent can't
  get tests passing within its repair-attempt cap, the workspace is
  rolled back to the pre-task commit rather than left half-done.
- **Check `logs/runs.jsonl`** for a structured record of any run
  (turns, tool calls, token usage, test results) - useful when a task
  didn't do what you expected and you want to see what the agent
  actually tried.

## Running in Docker

Build the image:

    docker build -t coding-agent .

Run it with your workspace mounted and the API key passed at runtime
(never baked into the image):

    docker run --rm -it \
      -e GEMINI_API_KEY=your-key-here \
      -v "$(pwd)/workspace:/workspace" \
      coding-agent

Only the `workspace` directory is mounted into the container — never the
full host filesystem. The image already points the agent at `/workspace`
(`WORKSPACE_DIR` is baked in) and runs it as a non-root user, so no
extra flags are needed for either. During local development,
`--env-file .env` is a convenient stand-in for `-e GEMINI_API_KEY=...`.

On Windows with PowerShell, use:

    docker run --rm -it `
      -e GEMINI_API_KEY=your-key-here `
      -v "${PWD}\workspace:/workspace" `
      coding-agent

**Git Bash gotcha:** running the `$(pwd)/workspace:/workspace` form from
Git Bash silently mangles the path (MSYS auto-converts it), which
mounts an empty/wrong directory instead of your real `workspace/` -
the container still runs and looks successful, it just never touches
your actual files. Confirmed by hitting this directly: a task reported
success but nothing showed up on disk until the mount was fixed.
Prefix the command with `MSYS_NO_PATHCONV=1` to disable that
conversion:

    MSYS_NO_PATHCONV=1 docker run --rm -it \
      -e GEMINI_API_KEY=your-key-here \
      -v "$(pwd)/workspace:/workspace" \
      coding-agent

## Security concept

The agent treats `workspace/` as the only place it is allowed to
operate. `sandbox.py`'s `resolve_path()` is the single choke point
every file tool goes through - it rejects absolute paths and any `..`
traversal that would resolve outside the workspace root, whether run
locally or via the `WORKSPACE_DIR`-mounted `/workspace` in Docker.

`run_command` is still a softer boundary than the file tools: it fixes
the subprocess's working directory to the workspace root and applies a
timeout, and now also strips known secrets (`GEMINI_API_KEY`) from the
subprocess's environment and caps captured output at 20,000 characters
- but a command can still reach outside the workspace path-wise (e.g.
`cat ../secret.txt`) at the shell level, since that's not something a
cwd restriction alone can prevent. Running **locally** (not in Docker),
it also still executes on the host's system `PATH`, meaning the agent
can install packages into your real global Python environment, not a
scoped one - real containment there requires actually using Docker.

Running in the container this project's `Dockerfile` builds closes
most of that gap: the agent runs as a non-root user (not root), inside
a disposable filesystem where "outside the workspace" is just the
container's own throwaway files, not your real machine, and pytest/
pytest-cov are pre-installed at build time so the agent's self-install
fallback (needed locally) is rarely exercised as non-root. Confirmed
by actually building and running the image, not just writing the
Dockerfile: non-root execution, all modules importing correctly
(catching a stale `COPY` list that would have broken the container
outright), the bind-mounted `/workspace` being writable by the non-root
user, and - after hitting and fixing git's real "dubious ownership"
refusal on a bind-mounted, differently-owned directory - a full task
running end to end (plan → code → test → coverage → review →
git-checkpoint finalize) inside the hardened container against the
real API.

Every task also gets git checkpoints (`git_manager.py`) in a nested
repo rooted at `workspace/` itself, separate from this project's own
repo - the outer `.gitignore` excludes all of `workspace/` so the two
histories never mix. If a task fails to recover or a coverage fix
breaks the suite, the workspace is rolled back to the last known-good
commit rather than left broken.

## Development roadmap

- **Phase 0 — Project Foundation** (done): project scaffold, environment,
  dependencies, Docker scaffolding, CLI shell.
- **Phase 1 — Sandbox boundary** (done): workspace path validation,
  traversal protection in `sandbox.py`.
- **Phase 2 — Tools** (done): the core file/command tools in `tools.py`.
- **Phase 3 — Agent** (done): the agent loop, tool calling, and task
  execution in `agent.py`, against Gemini via the OpenAI Agents SDK.
- **Phase 4 — Self-correction** (done): independent test verification
  and a repair loop, not trusting the agent's own claim of success.
- **Phase 5 — Planning** (done): a read-only planning pass (`planner.py`)
  before any writing/editing/running happens.
- **Phase 6 — Codebase search** (done): `search_code` for navigating
  beyond files the agent already knows about.
- **Phase 7 — Automatic test generation** (done): a coverage check that
  forces test generation for completely untested files.
- **Phase 8 — Git integration** (done): checkpoint/rollback around each
  stage via `git_manager.py`, so a failed attempt doesn't leave the
  workspace broken.
- **Phase 9 — Code review** (done): a second agent (`reviewer.py`)
  reviewing the actual diff before a task is reported done.
- **Phase 10 — Reliability & observability** (done): structured audit
  logging (`observability.py`) and history trimming to bound token
  growth across retry loops.
- **Phase 11 — Progress interface** (done): a live checklist in `main.py`
  (`ProgressDisplay`) instead of a single final result, driven by plain
  stage-name callbacks from `agent.py`.
- **Phase 12 — Docker sandbox hardening** (done): non-root execution,
  secret-stripped subprocess environment, output caps, pre-installed
  test tooling to avoid runtime pip-install-as-non-root - closing the
  `run_command` gap noted above.
- **Phase 13 — Full pipeline**: all of the above composed end to end -
  effectively already demonstrated by every phase's real end-to-end
  test since Phase 3, rather than a separate build step.

V4 (optional extensions beyond the core project):

- **Multi-agent task decomposition** (done): `coordinator.py` splits a
  genuinely multi-part task into subtasks, each run through the full
  agent.py pipeline in turn.
- **Persistent memory** (done): `memory.py` distills durable facts from
  each completed task into `workspace/.agent_memory.json`, grounding
  future tasks' planning in it.
- **Cross-process workspace locking** (done): `lock.py` file-locks the
  workspace for a run's full duration, so a local run and a Docker run
  (or two of either) started against the same `workspace/` at once
  fail fast instead of racing on the same git history.

## License

[MIT](LICENSE)
