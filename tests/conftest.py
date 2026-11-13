from pathlib import Path

import pytest

from swe_agent.cli.main import SANDBOX_BUILD, sandbox_problem

# tests/fixtures/ holds sample repos for the agent to work on. Their tests are
# not ours (the toy repo's fail on purpose), so pytest must not collect them.
collect_ignore = ["fixtures"]


@pytest.fixture(autouse=True)
def _isolated_agent_home(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Point SWE_AGENT_HOME at an empty folder for every test.

    Without this a test could read the real ~/.swe-agent/env, find a real API key and
    make a paid call, or write trajectories into the real home.
    """
    home = tmp_path / "agent-home"
    monkeypatch.setenv("SWE_AGENT_HOME", str(home))
    return home


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the tests marked `docker` when there is no daemon or no sandbox image.

    The check is the one `swe-agent run` makes before it starts. It runs only when at
    least one such test was collected, so a plain `pytest tests/test_cli.py` never
    calls docker.
    """
    docker_items = [item for item in items if item.get_closest_marker("docker")]
    if not docker_items:
        return
    problem = sandbox_problem()
    if problem is None:
        return
    for item in docker_items:
        item.add_marker(pytest.mark.skip(reason=f"{problem}; build it with: {SANDBOX_BUILD}"))
