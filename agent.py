"""Agent definition (the "brain").

Wires the five tools in tools.py to an OpenAI Agents SDK Agent,
running against Gemini via its OpenAI-compatible endpoint. Does not
talk to sandbox.py directly - that boundary belongs to tools.py.
"""

from __future__ import annotations

import os

from agents import (
    Agent,
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    retry_policies,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

import git_manager
import memory
import observability
import planner
import reviewer
import tools

load_dotenv()

# Tracing export defaults to reading OPENAI_API_KEY, which we don't
# set - disable it once here rather than let every run warn about it.
set_tracing_disabled(True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

MAX_AGENT_TURNS = 30
MAX_REPAIR_ATTEMPTS = 5
MAX_COVERAGE_ATTEMPTS = 2
MAX_REVIEW_ATTEMPTS = 2

# Gemini's free tier has told us directly (real 429s hit in testing)
# that it wants retries after 7-16s, not the SDK's ~2s default max
# backoff. provider_suggested() reads the provider's actual
# Retry-After advice when given; the backoff below is only the
# fallback for when no explicit advice is present.
RETRY_MODEL_SETTINGS = ModelSettings(
    retry=ModelRetrySettings(
        max_retries=3,
        backoff=ModelRetryBackoffSettings(initial_delay=2.0, max_delay=20.0, multiplier=2.0, jitter=True),
        policy=retry_policies.provider_suggested(),
    )
)

INSTRUCTIONS = """
You are an autonomous coding agent.

You operate exclusively inside /workspace.

You will be given a task and a suggested plan. Follow the plan, but
adapt it if inspecting the workspace shows it's wrong or incomplete.

Inspect before modifying.
Make minimal changes.
Run relevant tests.
Never claim success without verification.
""".strip()


def build_model() -> OpenAIChatCompletionsModel:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def build_agent(model: OpenAIChatCompletionsModel) -> Agent:
    agent_tools = [
        function_tool(tools.list_directory),
        function_tool(tools.read_file),
        function_tool(tools.search_code),
        function_tool(tools.write_file),
        function_tool(tools.edit_file),
        function_tool(tools.run_command),
    ]

    return Agent(
        name="Coding Agent",
        instructions=INSTRUCTIONS,
        model=model,
        model_settings=RETRY_MODEL_SETTINGS,
        tools=agent_tools,
    )


def _format_plan(task: str, steps: list[str]) -> str:
    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    return f"{task}\n\nSuggested plan:\n{numbered}"


def _short(text: str, limit: int = 60) -> str:
    text = text.strip().splitlines()[0] if text.strip() else text
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _emit(on_progress, label: str | None) -> None:
    """Report a stage label via on_progress, if one was given.

    on_progress is a plain callback - agent.py has no idea whether or
    how it gets displayed, that's entirely main.py's concern. A label
    of None signals the task is finished (whatever the outcome).
    """
    if on_progress is not None:
        on_progress(label)


def _finish(record, on_record, outcome: str, verified: bool | None) -> None:
    """Finish record, and report it via on_record, if one was given.

    Mirrors _emit for on_progress: on_record is a plain callback with
    no display purpose - coordinator.py uses it to get structured
    success/failure out of a subtask's run_task() call, rather than
    string-sniffing the human-readable return value for 'WARNING'/
    'NOTE'.
    """
    record_dict = record.finish(outcome, verified)
    if on_record is not None:
        on_record(record_dict)


def _trim_history(items: list, keep_last: int = 40) -> list:
    """Cap conversation history passed into a loop's next turn.

    Keeps the first item (the original task+plan) plus only the most
    recent keep_last items, dropping older tool-call/output pairs from
    the middle - a repair/coverage/review loop doesn't need a full
    transcript to keep making progress, just recent context plus what
    was originally asked. Bounds token growth across the up to ~10
    Runner.run_sync calls one task can involve.
    """
    if len(items) <= keep_last + 1:
        return items
    return [items[0]] + items[-keep_last:]


def run_task(
    task: str,
    max_turns: int = MAX_AGENT_TURNS,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    max_coverage_attempts: int = MAX_COVERAGE_ATTEMPTS,
    max_review_attempts: int = MAX_REVIEW_ATTEMPTS,
    on_progress=None,
    on_record=None,
) -> str:
    """Run a task to completion. See _run_task_inner for the full flow.

    Thin wrapper only: guarantees _emit(on_progress, None) - the "task
    is finished" signal - fires exactly once no matter which of
    _run_task_inner's several return points was hit, without needing
    every one of them to remember to signal completion individually.

    on_progress(label: str | None), if given, is called with a plain
    stage-name string at each stage boundary, and with None when the
    task finishes. agent.py doesn't know or care whether/how these are
    displayed - see main.py's ProgressDisplay for that.

    on_record(record: dict), if given, is called once with the
    structured observability record right as the task finishes -
    coordinator.py uses this to get real success/failure out of a
    subtask's run without string-sniffing the return value.

    Before returning, distills any durable facts worth remembering
    from this task (memory.py) into workspace/.agent_memory.json, so
    planner.py's future tasks in this project are grounded in it.
    Best-effort: a failure here (e.g. a rate limit) does not fail the
    task itself, which has already succeeded or failed on its own
    terms by this point.
    """
    try:
        result = _run_task_inner(
            task,
            max_turns,
            max_repair_attempts,
            max_coverage_attempts,
            max_review_attempts,
            on_progress,
            on_record,
        )
        _emit(on_progress, "Recording project memory")
        _save_task_memory(task, result)
        return result
    finally:
        _emit(on_progress, None)


def _save_task_memory(task: str, result: str) -> None:
    try:
        model = build_model()
        facts = memory.extract_memories(task, result, model, model_settings=RETRY_MODEL_SETTINGS)
        memory.save_memories(facts, task)
    except Exception:
        pass


def _run_task_inner(
    task: str,
    max_turns: int,
    max_repair_attempts: int,
    max_coverage_attempts: int,
    max_review_attempts: int,
    on_progress,
    on_record,
) -> str:
    """Plan, then run task to completion, self-correcting on failures and gaps.

    A planning pass (planner.py) inspects the workspace read-only and
    produces a short step list before any writing/editing/running
    happens. The coding agent is not bound to the plan - it can adapt
    if inspection shows it's wrong - but starts from it rather than a
    blank task description.

    The agent's own claim of success is not trusted: after each run,
    the test suite is executed independently (tools.run_tests). If it
    is failing, the failure output is fed back to the agent as a new
    turn and it gets another attempt, up to max_repair_attempts. A
    task with no tests yet ('no_tests') is not treated as a failure.

    Once tests pass, tools.check_coverage() looks for source files
    left at 0% coverage. If any exist, the agent gets up to
    max_coverage_attempts more turns to add tests for them - each
    followed by re-verifying tests still pass before checking coverage
    again, so filling a gap can't silently reintroduce a failure.

    git_manager checkpoints the workspace before any edits happen. If
    the repair loop exhausts its attempts still failing, or a coverage
    fill breaks the suite, the workspace is rolled back to the last
    known-good checkpoint instead of being left broken on disk.

    Finally, reviewer.py looks at the diff since the pre-task baseline
    (correctness, security, tests, maintainability, unrelated changes,
    regressions) and returns a verdict. If not approved, the issues
    are fed back for up to max_review_attempts more fixes, each
    re-verified against tests the same way. If it's still not approved
    after that, the result is still returned (it's working code, not
    broken code - review found quality concerns, not failures) but
    with the unresolved issues surfaced, not hidden. A clean approval
    ends in a final descriptive commit.

    Throughout, observability.RunRecord accumulates turns/tool-calls/
    token usage/attempt counts across every Runner.run_sync call, and
    is written to logs/runs.jsonl once at the end - independent of
    whether the task succeeded, so failures are as auditable as
    successes. Conversation history handed into each loop iteration is
    capped (_trim_history) rather than left to grow unbounded across
    the up to ~10 model calls a single task can involve.
    """
    record = observability.RunRecord(task=task)
    baseline = git_manager.checkpoint(f"baseline before task: {_short(task)}")

    _emit(on_progress, "Analyzing repository & planning")
    model = build_model()
    memory_context = memory.format_memories(memory.load_memories())
    planning_input = f"{memory_context}\n\n{task}" if memory_context else task
    steps = planner.create_plan(planning_input, model, model_settings=RETRY_MODEL_SETTINGS)
    task_with_plan = _format_plan(task, steps)

    _emit(on_progress, "Writing code")
    coding_agent = build_agent(model)
    result = Runner.run_sync(coding_agent, task_with_plan, max_turns=max_turns)
    record.add_result(result)

    _emit(on_progress, "Running tests")
    status, output = tools.run_tests()
    counts = tools.parse_test_counts(output)
    record.tests_passed, record.tests_failed = counts["passed"], counts["failed"]
    if status == "no_tests":
        _finish(record, on_record, "success_no_tests", None)
        return result.final_output

    attempt = 0
    while status == "failed" and attempt < max_repair_attempts:
        attempt += 1
        record.repair_attempts = attempt
        _emit(on_progress, f"Fixing (attempt {attempt}/{max_repair_attempts})")
        repair_prompt = (
            f"The test suite is still failing (repair attempt {attempt}/{max_repair_attempts}). "
            "Analyze the failure below, fix it, and make only the changes needed to "
            f"pass the tests.\n\n{output}"
        )
        history = _trim_history(result.to_input_list())
        result = Runner.run_sync(
            coding_agent,
            history + [{"role": "user", "content": repair_prompt}],
            max_turns=max_turns,
        )
        record.add_result(result)
        _emit(on_progress, "Running tests")
        status, output = tools.run_tests()
        counts = tools.parse_test_counts(output)
        record.tests_passed, record.tests_failed = counts["passed"], counts["failed"]

    if status == "failed":
        if baseline:
            git_manager.rollback(baseline)
        record.rolled_back = "repair attempts exhausted, still failing"
        _finish(record, on_record, "failed", False)
        return (
            f"{result.final_output}\n\n"
            f"WARNING: tests still failing after {max_repair_attempts} repair attempt(s). "
            f"Workspace rolled back to the state before this task.\n{output}"
        )

    # status == "passed" here - a known-good state worth being able
    # to return to before letting the coverage loop touch anything.
    last_good = git_manager.checkpoint("tests passing")

    _emit(on_progress, "Checking test coverage")
    coverage_status, coverage_output = tools.check_coverage()

    coverage_attempt = 0
    while coverage_status == "gaps" and coverage_attempt < max_coverage_attempts:
        coverage_attempt += 1
        record.coverage_attempts = coverage_attempt
        _emit(on_progress, f"Adding tests for coverage (attempt {coverage_attempt}/{max_coverage_attempts})")
        coverage_prompt = (
            f"Tests pass, but coverage shows untested source file(s) "
            f"(coverage attempt {coverage_attempt}/{max_coverage_attempts}). Add tests for "
            f"the file(s) at 0% coverage below. Do not modify already-tested code.\n\n{coverage_output}"
        )
        history = _trim_history(result.to_input_list())
        result = Runner.run_sync(
            coding_agent,
            history + [{"role": "user", "content": coverage_prompt}],
            max_turns=max_turns,
        )
        record.add_result(result)

        _emit(on_progress, "Running tests")
        test_status, test_output = tools.run_tests()
        if test_status == "failed":
            if last_good:
                git_manager.rollback(last_good)
            record.rolled_back = "coverage fill broke the test suite"
            _finish(record, on_record, "failed", False)
            return (
                f"{result.final_output}\n\n"
                f"WARNING: adding coverage broke the test suite. "
                f"Workspace rolled back to the last passing state.\n{test_output}"
            )
        counts = tools.parse_test_counts(test_output)
        record.tests_passed, record.tests_failed = counts["passed"], counts["failed"]

        _emit(on_progress, "Checking test coverage")
        coverage_status, coverage_output = tools.check_coverage()

    # A fresh known-good checkpoint right before review - the review
    # loop's rollback target if a review-driven fix breaks something.
    good_before_review = git_manager.checkpoint("pre-review checkpoint")

    diff = git_manager.get_diff(since=baseline) if baseline else ""
    if not diff.strip():
        _emit(on_progress, "Finalizing")
        git_manager.commit(f"Task: {_short(task)}")
        _finish(record, on_record, "success", True)
        return result.final_output

    _emit(on_progress, "Reviewing changes")
    verdict = reviewer.review_changes(task, diff, model, model_settings=RETRY_MODEL_SETTINGS)

    review_attempt = 0
    while not verdict.approved and review_attempt < max_review_attempts:
        review_attempt += 1
        record.review_attempts = review_attempt
        _emit(on_progress, f"Addressing review feedback (attempt {review_attempt}/{max_review_attempts})")
        issues = "\n".join(f"- {issue}" for issue in verdict.issues)
        review_prompt = (
            f"Code review found issues (review attempt {review_attempt}/{max_review_attempts}). "
            f"Address them, then the change will be re-reviewed.\n\n{issues}"
        )
        history = _trim_history(result.to_input_list())
        result = Runner.run_sync(
            coding_agent,
            history + [{"role": "user", "content": review_prompt}],
            max_turns=max_turns,
        )
        record.add_result(result)

        _emit(on_progress, "Running tests")
        test_status, test_output = tools.run_tests()
        if test_status == "failed":
            if good_before_review:
                git_manager.rollback(good_before_review)
            record.rolled_back = "review-driven fix broke the test suite"
            _finish(record, on_record, "failed", False)
            return (
                f"{result.final_output}\n\n"
                f"WARNING: addressing review feedback broke the test suite. "
                f"Workspace rolled back to the last passing state.\n{test_output}"
            )
        counts = tools.parse_test_counts(test_output)
        record.tests_passed, record.tests_failed = counts["passed"], counts["failed"]

        good_before_review = git_manager.checkpoint("pre-review checkpoint")
        diff = git_manager.get_diff(since=baseline)
        _emit(on_progress, "Reviewing changes")
        verdict = reviewer.review_changes(task, diff, model, model_settings=RETRY_MODEL_SETTINGS)

    _emit(on_progress, "Finalizing")
    if not verdict.approved:
        git_manager.commit(f"Task: {_short(task)} (unresolved review feedback)")
        _finish(record, on_record, "success_unresolved_review", True)
        issues = "\n".join(f"- {issue}" for issue in verdict.issues)
        return (
            f"{result.final_output}\n\n"
            f"NOTE: code review has unresolved concerns after {max_review_attempts} "
            f"attempt(s) - tests still pass, but review flagged:\n{issues}"
        )

    git_manager.commit(f"Task: {_short(task)}")
    _finish(record, on_record, "success", True)
    return result.final_output
