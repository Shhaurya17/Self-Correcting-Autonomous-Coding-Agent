"""CLI entry point.

Owns user interaction: displaying the heading, taking a task from
the user, showing live progress, and showing the final result. Holds
no agent reasoning logic itself - that lives in agent.py, reached via
coordinator.py (which decides whether a task should be decomposed
into subtasks first, then delegates to agent.py either way).
coordinator.py only calls a plain callback with stage-name strings;
it has no idea whether or how they get displayed. ProgressDisplay is
entirely this module's concern.
"""

import sys

import coordinator


def _pick_symbols() -> dict[str, str]:
    """Unicode checklist symbols if stdout's encoding can render them.

    Piped/redirected output on Windows commonly lands on a legacy
    codepage (e.g. cp1252) that raises UnicodeEncodeError on '✓'/'⟳' -
    checked once up front so the run degrades to ASCII instead of
    crashing partway through a task.
    """
    encoding = sys.stdout.encoding or "utf-8"
    try:
        "✓⟳".encode(encoding)
        return {"done": "✓", "active": "⟳"}
    except (UnicodeEncodeError, LookupError):
        return {"done": "[done]", "active": "[..]"}


class ProgressDisplay:
    """Renders agent.py's stage labels as a live checklist.

    Falls back to plain sequential lines (no cursor movement) when
    stdout isn't a terminal - e.g. output piped or redirected - so
    this never prints raw escape codes into a file or another tool's
    input.
    """

    def __init__(self) -> None:
        self._done: list[str] = []
        self._active: str | None = None
        self._live = sys.stdout.isatty()
        self._last_render_lines = 0
        self._symbols = _pick_symbols()

    def update(self, label: str | None) -> None:
        if self._active is not None:
            self._done.append(self._active)
            if not self._live:
                print(f"  {self._symbols['done']} {self._active}")
        self._active = label

        if self._live:
            self._render()
        elif label is not None:
            print(f"  {self._symbols['active']} {label}...")

    def _render(self) -> None:
        if self._last_render_lines:
            sys.stdout.write(f"\033[{self._last_render_lines}A\033[J")

        lines = [f"  {self._symbols['done']} {step}" for step in self._done]
        if self._active is not None:
            lines.append(f"  {self._symbols['active']} {self._active}...")

        for line in lines:
            print(line)
        sys.stdout.flush()
        self._last_render_lines = len(lines)


def main() -> None:
    print("=" * 32)
    print(" Autonomous Coding Agent")
    print("=" * 32)
    print()
    print("Task:")

    task = input("> ").strip()

    if not task:
        print("No task entered. Exiting.")
        return

    print()
    progress = ProgressDisplay()
    try:
        result = coordinator.run_task(task, on_progress=progress.update)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    print()
    print(result)


if __name__ == "__main__":
    main()
