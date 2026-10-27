"""The `bash` tool: run one command in a fresh shell, with a timeout and capped output."""

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel

from swe_agent.models.types import ToolSpec

BASH_SPEC = ToolSpec(
    name="bash",
    description=(
        "Run a command in bash, starting from the repository root. "
        "Each call is a fresh shell: `cd` and exported variables do not persist. "
        "Returns the exit code and the combined stdout and stderr. "
        "Commands are killed after 30 seconds. Long output is cut in the middle."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run, e.g. `ls src`"},
        },
        "required": ["command"],
    },
)


class BashResult(BaseModel):
    command: str
    output: str  # stdout and stderr combined, after truncation
    exit_code: int | None  # None when the command was killed by the timeout
    timed_out: bool
    truncated_chars: int
    started_ns: int
    ended_ns: int


def truncate_middle(text: str, max_chars: int) -> tuple[str, int]:
    """Keep the head and tail of `text`; return (text, number of characters cut).

    Short text comes back unchanged with 0. Otherwise keep max_chars // 2 characters from
    the start and the rest from the end, joined by "\\n[... N characters omitted ...]\\n".
    """
    if max_chars <= 0:
        return ("", len(text))
    if len(text) <= max_chars:
        return (text, 0)
    head = max_chars // 2
    tail = max_chars - head
    truncated_chars = len(text) - max_chars
    output = text[:head] + f"\n[... {truncated_chars} characters omitted ...]\n" + text[-tail:]
    return (output, truncated_chars)


def run_bash(
    command: str, cwd: Path, *, timeout_s: float = 30.0, max_output_chars: int = 10_000
) -> BashResult:
    """Run `command` with /bin/bash -c in `cwd`.

    Record time.monotonic_ns() before and after. Start the process exactly as in the
    "Process groups" snippet in 04-bash-tool.md (stdin closed, stderr merged into stdout,
    its own session). On timeout kill the whole process group. Decode the bytes with
    errors="replace", then truncate_middle.
    """
    is_timeout = False
    started_ns = time.monotonic_ns()

    proc = subprocess.Popen(
        ["/bin/bash", "-c", command],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        raw, _ = proc.communicate(timeout=timeout_s)
        ended_ns = time.monotonic_ns()
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        raw, _ = proc.communicate()
        ended_ns = time.monotonic_ns()
        is_timeout = True

    code = proc.returncode
    if is_timeout:
        code = None
    output, truncate_chars = truncate_middle(raw.decode(errors="replace"), max_output_chars)

    return BashResult(
        command=command,
        output=output,
        exit_code=code,
        timed_out=is_timeout,
        truncated_chars=truncate_chars,
        started_ns=started_ns,
        ended_ns=ended_ns,
    )


def format_observation(result: BashResult) -> str:
    """The text the model sees for one command:

        exit code: 1
        <output>
        ...output...
        </output>

    On timeout, replace the first line with
    "timed out; the command and its child processes were killed".
    """
    str = ""
    if result.timed_out:
        str += "timed out; the command and its child processes were killed"
    else:
        str += f"exit code: {result.exit_code}\n"
    str += f"{result.output}\n"
    return str
