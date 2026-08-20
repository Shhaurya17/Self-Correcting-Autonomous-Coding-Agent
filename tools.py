"""Agent tools.

The tools the agent uses to interact with the workspace.
list_directory/read_file/write_file/edit_file/search_code resolve
every path through sandbox.py, so they cannot touch anything outside
workspace/.

run_command fixes the working directory to the workspace root,
strips secrets from the subprocess's environment, caps captured
output, and applies a timeout - it still does not stop a command
from reaching outside the workspace (e.g. `cat ../../secret.txt`).
Full process isolation (no access outside workspace at all, non-root
execution) comes from running inside the Docker container built by
this project's Dockerfile, not from anything in this module alone.

run_tests/check_coverage are not agent-facing tools - not wrapped
with function_tool or exposed to the LLM. agent.py's orchestration
calls them directly to check results independently of whatever the
agent itself claims.
"""

from __future__ import annotations

import os
import re
import subprocess

from sandbox import (
    get_workspace_root,
    resolve_existing_dir,
    resolve_existing_file,
    resolve_path,
)

# Denylist, not allowlist: subprocess (especially shell=True on
# Windows) needs PATH/SYSTEMROOT/COMSPEC/etc. to function at all, and
# enumerating everything required per-platform is fragile. Stripping
# the few keys we know are secrets is safer and portable than trying
# to reconstruct a minimal environment from scratch.
_SENSITIVE_ENV_VARS = {"GEMINI_API_KEY", "OPENAI_API_KEY"}

# Cap on captured stdout/stderr per subprocess call - a runaway or
# just chatty command shouldn't be able to blow up token usage or
# memory. ~5000 tokens' worth of characters, generous for real test
# output.
_MAX_OUTPUT_CHARS = 20_000


def _sanitized_env() -> dict[str, str]:
    """A copy of the current environment with known secrets stripped."""
    env = os.environ.copy()
    for key in _SENSITIVE_ENV_VARS:
        env.pop(key, None)
    return env


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated at {limit} characters"


def list_directory(relative_path: str = ".") -> list[str]:
    """List the immediate contents of a directory in the workspace."""
    directory = resolve_existing_dir(relative_path)
    entries = []
    for entry in sorted(directory.iterdir()):
        suffix = "/" if entry.is_dir() else ""
        entries.append(f"{entry.name}{suffix}")
    return entries


def read_file(relative_path: str) -> str:
    """Read the full text contents of a file in the workspace."""
    path = resolve_existing_file(relative_path)
    return path.read_text(encoding="utf-8")


def write_file(relative_path: str, content: str) -> str:
    """Create or overwrite a file in the workspace, creating parent dirs."""
    path = resolve_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {relative_path}"


def search_code(query: str, case_sensitive: bool = False, max_results: int = 200) -> list[str]:
    """Search all files in the workspace for lines containing query.

    query is matched as a plain substring, not a regex. Skips hidden
    directories (name starts with '.') and __pycache__, and silently
    skips files that aren't decodable as UTF-8 text (binaries).
    Returns up to max_results 'relative/path:line: content' entries,
    plus a final truncation note if the cap was hit.
    """
    root = get_workspace_root()
    needle = query if case_sensitive else query.lower()
    matches: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                matches.append(f"{rel}:{lineno}: {line.strip()}")
                if len(matches) >= max_results:
                    matches.append(f"... truncated at {max_results} results")
                    return matches

    return matches


def edit_file(relative_path: str, old_text: str, new_text: str) -> str:
    """Replace an exact, unique occurrence of old_text with new_text.

    Raises ValueError if old_text has zero matches or more than one
    match in the file - an edit must be unambiguous.
    """
    path = resolve_existing_file(relative_path)
    content = path.read_text(encoding="utf-8")

    count = content.count(old_text)
    if count == 0:
        raise ValueError(f"No match for old_text in {relative_path!r}")
    if count > 1:
        raise ValueError(f"old_text is not unique in {relative_path!r} ({count} matches)")

    new_content = content.replace(old_text, new_text, 1)
    path.write_text(new_content, encoding="utf-8")
    return f"Edited {relative_path}"


def run_command(command: str, timeout: int = 120) -> str:
    """Run a shell command with its cwd fixed to the workspace root.

    Confinement is still only convenience-level - see module
    docstring - but the subprocess environment has known secrets
    stripped, and captured output is capped.
    """
    root = get_workspace_root()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=root,
            env=_sanitized_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s: {command!r}"

    output = f"$ {command}\nexit code: {result.returncode}\n"
    if result.stdout:
        output += f"--- stdout ---\n{_truncate(result.stdout)}"
    if result.stderr:
        output += f"--- stderr ---\n{_truncate(result.stderr)}"
    return output


def run_tests(command: str = "python -m pytest -q", timeout: int = 120) -> tuple[str, str]:
    """Run the test suite and report its outcome independently of the agent.

    Returns (status, output) where status is 'passed', 'failed', or
    'no_tests' (pytest's exit code 5: nothing was collected - a task
    with no tests yet is not a failure to repair).
    """
    root = get_workspace_root()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=root,
            env=_sanitized_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "failed", f"Test command timed out after {timeout}s: {command!r}"

    output = f"$ {command}\nexit code: {result.returncode}\n{_truncate(result.stdout + result.stderr)}"

    if result.returncode == 0:
        return "passed", output
    if result.returncode == 5:
        return "no_tests", output
    return "failed", output


_PASSED_COUNT = re.compile(r"(\d+) passed")
_FAILED_COUNT = re.compile(r"(\d+) failed")


def parse_test_counts(output: str) -> dict[str, int]:
    """Pull passed/failed counts out of a pytest summary line, for logging.

    Best-effort text parsing, not a substitute for the pass/fail
    status from run_tests()/check_coverage() - a count of 0 here can
    mean "no tests of that kind" just as easily as "couldn't parse".
    """
    passed = _PASSED_COUNT.search(output)
    failed = _FAILED_COUNT.search(output)
    return {
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
    }


_COVERAGE_ROW = re.compile(r"^(\S+\.py)\s+(\d+)\s+(\d+)\s+(\d+)%")


def check_coverage(
    command: str = "python -m pytest --cov=. --cov-report=term-missing -q",
    timeout: int = 120,
) -> tuple[str, str]:
    """Run the test suite with coverage measurement.

    Auto-installs pytest-cov (into whatever environment run_command
    itself uses) if the --cov flag isn't recognized yet, mirroring how
    the agent has always had to self-install pytest.

    Returns (status, output) where status is:
      'failed'      - the tests themselves are failing (not this
                       function's concern; the repair loop owns that)
      'no_tests'    - nothing to run yet
      'unavailable' - pytest-cov could not be installed
      'gaps'        - tests pass, but at least one non-test source
                       file has 0% coverage
      'passed'      - tests pass and no source file is at 0%
    """
    root = get_workspace_root()

    def _run(cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, shell=True, cwd=root, env=_sanitized_env(), capture_output=True, text=True, timeout=timeout
        )

    try:
        result = _run(command)
    except subprocess.TimeoutExpired:
        return "failed", f"Test command timed out after {timeout}s: {command!r}"

    if "unrecognized arguments" in result.stderr and "--cov" in result.stderr:
        install = _run("python -m pip install pytest-cov")
        if install.returncode != 0:
            return "unavailable", _truncate(install.stdout + install.stderr)
        try:
            result = _run(command)
        except subprocess.TimeoutExpired:
            return "failed", f"Test command timed out after {timeout}s: {command!r}"

    # Truncate only the string handed back to the caller - coverage-row
    # parsing below still reads the full, untruncated result.stdout.
    output = f"$ {command}\nexit code: {result.returncode}\n{_truncate(result.stdout + result.stderr)}"

    if result.returncode == 5:
        return "no_tests", output
    if result.returncode != 0:
        return "failed", output

    zero_coverage_files = []
    for line in result.stdout.splitlines():
        match = _COVERAGE_ROW.match(line)
        if not match:
            continue
        filename = match.group(1)
        percent = int(match.group(4))
        is_test_file = filename.startswith("test_") or "/test_" in filename or "\\test_" in filename
        if percent == 0 and not is_test_file:
            zero_coverage_files.append(filename)

    if zero_coverage_files:
        return "gaps", output

    return "passed", output
