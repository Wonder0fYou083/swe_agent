"""The sandbox image has what the agent's commands expect.

These tests start real containers, so they need Docker and the image (see conftest.py).
They call `docker run` directly: they check the image, not the bash tool.
"""

import subprocess

import pytest

from swe_agent.tools.bash import SANDBOX_IMAGE

pytestmark = pytest.mark.docker


def run_in_image(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "run", "--rm", "--network", "none", SANDBOX_IMAGE, "bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )


def test_image_has_python_3_12_and_the_locked_pytest() -> None:
    result = run_in_image("python --version && python -m pytest --version")
    assert result.returncode == 0, result.stdout
    assert "Python 3.12." in result.stdout
    assert "pytest 9.1.1" in result.stdout


def test_image_has_gnu_tools_not_bsd() -> None:
    # The host is macOS with BSD tools, so the image must provide GNU ones for the
    # sandbox prompt to promise them.
    result = run_in_image("sed --version && grep --version")
    assert result.returncode == 0, result.stdout
    assert "GNU sed" in result.stdout
    assert "GNU grep" in result.stdout
