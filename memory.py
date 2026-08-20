"""Persistent project memory.

Durable facts/decisions worth carrying across separate tasks in the
same project - "the User model is in models.py", "chose JWT over
session cookies for auth" - distilled by a small LLM pass after each
task, not a raw log (that's what git history and observability.py's
runs.jsonl already are).

Stored inside workspace/ itself (.agent_memory.json), not alongside
this tool's own operational state (logs/) - it's knowledge ABOUT that
specific project, so it should travel with the workspace. Pointed at
a different project, a different project's memory should apply, not
this one's.

Not exposed to the agent as a tool. agent.py calls extract_memories()
after a task finishes (success or failure - a failed approach is
worth remembering too) and save_memories() to persist it, and injects
load_memories()/format_memories() into planner.py's task input before
planning. planner.py itself has no memory-specific code - it just
receives a task string with prior context already folded in.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, ModelSettings, Runner
from pydantic import BaseModel

from sandbox import get_workspace_root

MAX_LOADED_MEMORIES = 20

MEMORY_INSTRUCTIONS = """
You extract durable facts and decisions worth remembering from a
completed coding task, for grounding future tasks in the same
project.

Only extract things that would still be true and useful days from
now: where something lives ("the User model is in models.py"),
decisions made and why ("chose JWT over session cookies for auth -
simpler for a stateless API"), constraints discovered ("the payment
module can't be changed without breaking existing tests"), or
approaches that failed and shouldn't be repeated.

Do NOT extract routine implementation details that follow obviously
from the code, test counts, or anything that won't matter once this
task is done. Most tasks produce zero facts worth remembering - only
extract something if it would genuinely help someone (or another
agent) working on this project later. Return an empty list rather
than padding it with filler.
""".strip()


class ExtractedMemories(BaseModel):
    facts: list[str]


def _memory_path() -> Path:
    return get_workspace_root() / ".agent_memory.json"


def build_extractor(model, model_settings=None) -> Agent:
    return Agent(
        name="Memory Extractor",
        instructions=MEMORY_INSTRUCTIONS,
        model=model,
        model_settings=model_settings or ModelSettings(),
        output_type=ExtractedMemories,
    )


def extract_memories(task: str, result: str, model, model_settings=None, max_turns: int = 3) -> list[str]:
    """Distill durable facts worth remembering from a completed task."""
    extractor = build_extractor(model, model_settings)
    prompt = f"Task: {task}\n\nResult:\n{result}"
    run_result = Runner.run_sync(extractor, prompt, max_turns=max_turns)
    return run_result.final_output.facts


def _read_raw() -> list[dict]:
    path = _memory_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_memories(facts: list[str], task: str) -> None:
    """Append facts (if any) to the workspace's memory store."""
    if not facts:
        return
    entries = _read_raw()
    entries.append({"task": task, "facts": facts})
    _memory_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def load_memories(limit: int = MAX_LOADED_MEMORIES) -> list[dict]:
    """Return the most recent memory entries, oldest first."""
    return _read_raw()[-limit:]


def format_memories(entries: list[dict]) -> str:
    """Render memory entries as a compact block for prompt injection."""
    if not entries:
        return ""
    lines = [fact for entry in entries for fact in entry["facts"]]
    return "Known context from prior tasks in this project:\n" + "\n".join(f"- {line}" for line in lines)
