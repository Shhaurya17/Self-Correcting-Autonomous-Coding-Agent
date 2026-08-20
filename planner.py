"""Planning agent.

Produces a short, ordered plan for a task before the coding agent
starts acting. Grounded in the real workspace via read-only tools
(list_directory, read_file, search_code) - it cannot write, edit, or
run commands. Does not import agent.py; agent.py imports this and
supplies the already-built model, so there is no import cycle.
"""

from __future__ import annotations

from agents import Agent, ModelSettings, Runner, function_tool
from pydantic import BaseModel

import tools

PLANNER_INSTRUCTIONS = """
You are the planning stage of an autonomous coding agent.

Given a task, inspect the workspace as needed - you may list
directories, search the codebase for relevant files, and read files,
but you cannot write, edit, or run commands. Produce a short,
concrete, ordered list of the steps the coding agent should take to
complete this specific task. Keep it tight: only the steps actually
needed here, not a generic checklist.
""".strip()


class Plan(BaseModel):
    steps: list[str]


def build_planner(model, model_settings=None) -> Agent:
    return Agent(
        name="Planner",
        instructions=PLANNER_INSTRUCTIONS,
        model=model,
        model_settings=model_settings or ModelSettings(),
        tools=[
            function_tool(tools.list_directory),
            function_tool(tools.read_file),
            function_tool(tools.search_code),
        ],
        output_type=Plan,
    )


def create_plan(task: str, model, model_settings=None, max_turns: int = 10) -> list[str]:
    """Produce an ordered list of steps for task, grounded in workspace state."""
    planner = build_planner(model, model_settings)
    result = Runner.run_sync(planner, task, max_turns=max_turns)
    plan: Plan = result.final_output
    return plan.steps
