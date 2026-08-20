"""Coordinator: task decomposition and delegation.

Sits above agent.py, not inside it - for a task that's actually
several largely independent pieces, breaks it up and runs each
subtask through the existing agent.run_task() pipeline (full
plan->code->test->coverage->review->checkpoint cycle) in turn, rather
than asking one agent invocation to do everything at once. For an
atomic task (the common case), this is a thin pass-through: one extra
decision call, then agent.run_task() unchanged.

Subtasks run sequentially against the same workspace/git history - no
parallel-write conflicts to solve, since only one subtask's coding
agent ever touches the filesystem at a time, and each subtask's own
planning step naturally sees prior subtasks' committed work.
"""

from __future__ import annotations

from agents import Agent, ModelSettings, Runner, function_tool
from pydantic import BaseModel

import agent as agent_module
import lock
import tools

# A decomposition proposing more than this is treated as evidence the
# split itself is poorly scoped (or the task doesn't actually warrant
# decomposition), not something to blindly chase - each subtask can
# already involve up to ~10 model calls of its own via agent.py's
# repair/coverage/review loops.
MAX_SUBTASKS = 5

COORDINATOR_INSTRUCTIONS = """
You are the coordination stage of an autonomous coding agent.

Given a task, decide whether it is actually several largely
independent pieces of work that would go better as separate,
sequential subtasks - each getting its own full implement-test-review
cycle - rather than one pass.

Most tasks are NOT like this: a single function, a single bug fix, a
small feature are atomic and should not be decomposed. Only split a
task that is genuinely composed of multiple substantial, mostly
independent parts (e.g. "build a REST API with authentication, a
database layer, and a test suite" might reasonably split into
separate subtasks for each layer; "write a function to reverse a
string" never should).

If you do decompose, each subtask must be a complete, self-contained
task description usable entirely on its own - not a sentence
fragment - and subtasks should be ordered so later ones can build on
earlier ones' output (e.g. a data layer before the API that uses it).

You may inspect the workspace (list directories, search, read files)
to see what already exists before deciding.
""".strip()


class DecompositionResult(BaseModel):
    should_decompose: bool
    reasoning: str
    subtasks: list[str]


def _emit(on_progress, label: str | None) -> None:
    """Local copy of agent._emit - avoids reaching into agent.py's
    underscore-prefixed (module-private) helper from here."""
    if on_progress is not None:
        on_progress(label)


def build_coordinator(model, model_settings=None) -> Agent:
    return Agent(
        name="Coordinator",
        instructions=COORDINATOR_INSTRUCTIONS,
        model=model,
        model_settings=model_settings or ModelSettings(),
        tools=[
            function_tool(tools.list_directory),
            function_tool(tools.read_file),
            function_tool(tools.search_code),
        ],
        output_type=DecompositionResult,
    )


def decompose(task: str, model, model_settings=None, max_turns: int = 10) -> DecompositionResult:
    """Decide whether task should be split into subtasks, and into what."""
    coordinator = build_coordinator(model, model_settings)
    result = Runner.run_sync(coordinator, task, max_turns=max_turns)
    return result.final_output


def _wrap_progress(on_progress, prefix: str | None = None):
    """Wrap on_progress for one inner agent.run_task() call.

    Swallows that call's own finish signal (None) - coordinator.py's
    run_task() owns the single outer finish signal itself (see its
    own try/finally), emitted exactly once regardless of whether
    decomposition happened or how many subtasks ran. Without this, an
    atomic (pass-through) task would fire on_progress(None) twice -
    once from the inner call's own guarantee, once from the outer -
    and a decomposed task's live display would look finished after
    just the first subtask.
    """
    if on_progress is None:
        return None

    def wrapped(label: str | None) -> None:
        if label is not None:
            on_progress(f"{prefix}{label}" if prefix else label)

    return wrapped


def run_task(
    task: str,
    max_turns: int = agent_module.MAX_AGENT_TURNS,
    max_repair_attempts: int = agent_module.MAX_REPAIR_ATTEMPTS,
    max_coverage_attempts: int = agent_module.MAX_COVERAGE_ATTEMPTS,
    max_review_attempts: int = agent_module.MAX_REVIEW_ATTEMPTS,
    on_progress=None,
) -> str:
    """Run task, decomposing into sequential subtasks first if warranted.

    For an atomic task (the common case) this is a thin pass-through
    to agent.run_task(). For a genuinely multi-part task, each subtask
    gets its own full agent.run_task() pipeline run in turn against
    the same workspace/git history, and results are aggregated into
    one combined report. If a subtask is not verified (a real
    failure, not just an unresolved review note), later subtasks are
    not attempted rather than building on broken state.

    The whole call - decomposition decision plus every subtask - holds
    lock.held() for its full duration, not just each individual
    agent.run_task() call: a second run starting between two subtasks
    of this one would still race on the same workspace/git history, so
    the lock has to span the entire multi-part task, not each piece of
    it. Raises lock.WorkspaceLockedError (a RuntimeError) if another
    run already holds it - main.py already handles RuntimeError.
    """
    try:
        with lock.held():
            model = agent_module.build_model()
            _emit(on_progress, "Deciding whether to decompose the task")
            decision = decompose(task, model, model_settings=agent_module.RETRY_MODEL_SETTINGS)

            subtasks = decision.subtasks
            if not decision.should_decompose or not subtasks or len(subtasks) > MAX_SUBTASKS:
                return agent_module.run_task(
                    task,
                    max_turns=max_turns,
                    max_repair_attempts=max_repair_attempts,
                    max_coverage_attempts=max_coverage_attempts,
                    max_review_attempts=max_review_attempts,
                    on_progress=_wrap_progress(on_progress),
                )

            total = len(subtasks)
            reports = []
            all_verified = True
            for i, subtask in enumerate(subtasks, start=1):
                records: list[dict] = []
                summary = agent_module.run_task(
                    subtask,
                    max_turns=max_turns,
                    max_repair_attempts=max_repair_attempts,
                    max_coverage_attempts=max_coverage_attempts,
                    max_review_attempts=max_review_attempts,
                    on_progress=_wrap_progress(on_progress, prefix=f"[{i}/{total}] "),
                    on_record=records.append,
                )
                verified = records[-1].get("verified") if records else None
                reports.append(f"### Subtask {i}/{total}: {subtask}\n\n{summary}")

                if verified is False:
                    all_verified = False
                    reports.append(f"\nSTOPPED: subtask {i} failed; later subtasks were not attempted.")
                    break

            header = f"Task decomposed into {total} subtask(s). {decision.reasoning}\n\n"
            header += (
                "All subtasks verified successfully.\n"
                if all_verified
                else "One or more subtasks did not complete successfully.\n"
            )
            return header + "\n\n".join(reports)
    finally:
        _emit(on_progress, None)
