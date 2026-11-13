"""Escape attempts against the sandboxed bash tool.

Each attempt must fail cleanly, which means four things:

1. The tool returns a normal result (a non-zero exit code, or timed out) and nothing raises.
2. The call ends within the timeout plus a few seconds.
3. No container is left behind.
4. Nothing on the host outside the workspace changed.

`run_cleanly` checks the four rules on every call. Each test then checks how its attempt
failed. These tests start real containers, so they need Docker and the image (see conftest.py).
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from swe_agent.tools.bash import SANDBOX_IMAGE, BashResult, run_bash_in_sandbox

pytestmark = pytest.mark.docker

GRACE_S = 5.0  # "a few seconds": starting the container, and `docker rm -f` after a timeout
NEIGHBOUR_TEXT = "a host file next to the workspace\n"
HOST_ONLY_VALUE = "this-value-lives-only-on-the-host"


def sandbox_containers() -> set[str]:
    """The IDs of every container made from the sandbox image, running or stopped."""
    listing = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", f"ancestor={SANDBOX_IMAGE}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return set(listing.stdout.split())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace, with a host file right next to it that no command may touch."""
    (tmp_path / "neighbour.txt").write_text(NEIGHBOUR_TEXT)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def run_cleanly(command: str, workspace: Path, *, timeout_s: float = 30.0) -> BashResult:
    """Run `command` in the sandbox and check the four "fail cleanly" rules."""
    outside = workspace.parent
    entries_before = set(outside.iterdir())
    containers_before = sandbox_containers()

    started = time.monotonic()
    result = run_bash_in_sandbox(command, workspace, timeout_s=timeout_s)  # rule 1: no raise
    elapsed = time.monotonic() - started

    assert (result.exit_code is None) == result.timed_out  # rule 1: a normal result
    assert elapsed < timeout_s + GRACE_S, f"the call took {elapsed:.1f} s"  # rule 2
    assert sandbox_containers() <= containers_before, "a container was left behind"  # rule 3
    assert set(outside.iterdir()) == entries_before  # rule 4
    assert (outside / "neighbour.txt").read_text() == NEIGHBOUR_TEXT  # rule 4
    return result


@pytest.mark.parametrize("command", ["touch /etc/x", "touch /x", "touch ../x"])
def test_writing_outside_the_workspace_fails(workspace: Path, command: str) -> None:
    result = run_cleanly(command, workspace)

    assert result.exit_code != 0
    assert "Read-only file system" in result.output
    assert list(workspace.iterdir()) == []


def test_writing_inside_the_workspace_reaches_the_host(workspace: Path) -> None:
    result = run_cleanly("touch x", workspace)

    assert result.exit_code == 0
    # On Linux this also checks --user. Docker Desktop on macOS shows every file in the
    # mount as yours, even one root wrote, so the next test checks --user directly.
    assert (workspace / "x").stat().st_uid == os.getuid()


def test_commands_run_as_your_user_not_root(workspace: Path) -> None:
    result = run_cleanly("id -u", workspace)

    assert result.output.strip() == str(os.getuid())


def test_the_network_is_unreachable(workspace: Path) -> None:
    result = run_cleanly("echo > /dev/tcp/1.1.1.1/80", workspace)

    assert result.exit_code != 0
    assert "Network is unreachable" in result.output


def test_a_fork_bomb_is_capped_and_ends_by_timeout(workspace: Path) -> None:
    # Never fork-bomb the host: first make sure commands really run in a container.
    probe = run_cleanly("test -e /.dockerenv || test -e /run/.containerenv", workspace)
    assert probe.exit_code == 0, "the command did not run in a container; not starting the bomb"

    # The trailing sleep keeps the container's first process alive. Without it bash exits at
    # once, the container stops, and the test proves nothing.
    result = run_cleanly(":(){ :|:& };:; sleep 60", workspace, timeout_s=3)

    assert result.timed_out
    assert result.exit_code is None


def test_using_more_memory_than_the_cap_is_killed(workspace: Path) -> None:
    # Twice the 1 GiB cap. Building the string writes every byte, so the memory is really used.
    result = run_cleanly("python -c 'x = \"a\" * 2**31'", workspace)

    assert result.exit_code == 137  # 128 + 9: the OOM killer's SIGKILL


def test_a_command_past_its_timeout_is_stopped(workspace: Path) -> None:
    result = run_cleanly("sleep 60", workspace, timeout_s=1)

    assert result.timed_out
    assert result.exit_code is None


def test_host_secrets_do_not_reach_the_command(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", HOST_ONLY_VALUE)
    monkeypatch.setenv("OPENAI_API_KEY", HOST_ONLY_VALUE)

    result = run_cleanly("env", workspace)

    assert result.exit_code == 0
    assert HOST_ONLY_VALUE not in result.output
