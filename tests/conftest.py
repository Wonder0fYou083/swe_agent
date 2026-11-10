from pathlib import Path

import pytest

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
