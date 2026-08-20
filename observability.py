"""Structured audit logging for agent runs.

Not exposed to the agent as a tool, and doesn't judge anything the
way tools.run_tests/check_coverage or reviewer.py do - it just
records what happened. agent.py builds up one RunRecord across a
whole run_task() call and writes it out once at the end as a JSON
line, so the history of runs is auditable after the fact.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "logs" / "runs.jsonl"


@dataclass
class RunRecord:
    task: str
    started_at: float = field(default_factory=time.time)
    turns: int = 0
    tool_calls: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    repair_attempts: int = 0
    coverage_attempts: int = 0
    review_attempts: int = 0
    tests_passed: int | None = None
    tests_failed: int | None = None
    rolled_back: str | None = None
    outcome: str = "unknown"
    verified: bool | None = None
    duration_seconds: float = 0.0

    def add_result(self, result) -> None:
        """Fold in turns/tool-calls/usage from one Runner.run_sync result."""
        self.turns += len(result.raw_responses)
        self.tool_calls += sum(1 for item in result.new_items if type(item).__name__ == "ToolCallItem")

        usage = result.context_wrapper.usage
        self.requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def finish(self, outcome: str, verified: bool | None) -> dict:
        self.outcome = outcome
        self.verified = verified
        self.duration_seconds = round(time.time() - self.started_at, 2)
        record = asdict(self)
        write(record)
        return record


def write(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
