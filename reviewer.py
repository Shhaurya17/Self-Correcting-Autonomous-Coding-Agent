"""Review agent.

A second reasoning pass over the coding agent's own diff before a
task is reported done. Read-only (list_directory, read_file,
search_code) - it cannot write, edit, or run commands, and doesn't
re-run tests itself (agent.py's run_tests/check_coverage already own
that independently). Does not import agent.py; agent.py builds the
model, gets the diff from git_manager, and passes both in, so there
is no import cycle.
"""

from __future__ import annotations

from agents import Agent, ModelSettings, Runner, function_tool
from pydantic import BaseModel

import tools

REVIEWER_INSTRUCTIONS = """
You are the review stage of an autonomous coding agent, checking the
changes another agent just made before they're reported as done.

You will be given the task and a diff of the changes made for it.
Inspect the workspace as needed - you may list directories, search
the codebase, and read files, but you cannot write, edit, or run
commands.

Check the diff for:
- Correctness: does the logic actually do what the task needs?
- Security: injection, unsafe eval, path traversal, hardcoded secrets, etc.
- Tests: do they meaningfully cover the change, not just pad a count?
- Maintainability: is it reasonably clear, not needlessly complex?
- Unnecessary changes: anything unrelated to the task that shouldn't be there?
- Regressions: could this plausibly break something that worked before?

Be a real reviewer, not a rubber stamp: approve only if you would
actually be comfortable with this diff landing. If you find a
problem, describe it concretely enough that a fix is obvious - name
the file and what's wrong, not a vague concern.
""".strip()


class ReviewVerdict(BaseModel):
    approved: bool
    summary: str
    issues: list[str]


def build_reviewer(model, model_settings=None) -> Agent:
    return Agent(
        name="Reviewer",
        instructions=REVIEWER_INSTRUCTIONS,
        model=model,
        model_settings=model_settings or ModelSettings(),
        tools=[
            function_tool(tools.list_directory),
            function_tool(tools.read_file),
            function_tool(tools.search_code),
        ],
        output_type=ReviewVerdict,
    )


def review_changes(task: str, diff: str, model, model_settings=None, max_turns: int = 10) -> ReviewVerdict:
    """Review diff (made for task) and return a verdict."""
    reviewer = build_reviewer(model, model_settings)
    prompt = f"Task: {task}\n\nDiff of changes made for this task:\n\n{diff}"
    result = Runner.run_sync(reviewer, prompt, max_turns=max_turns)
    return result.final_output
