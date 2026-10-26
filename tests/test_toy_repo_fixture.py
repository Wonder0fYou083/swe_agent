"""Guards for the toy repo, the task behind Phase 1's `swe-agent run "make the tests pass"`.

The toy repo must keep exactly one failing test, and one small edit must fix it. Each test
works on a copy, so the fixture itself is never changed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOY_REPO = Path(__file__).parent / "fixtures" / "toy_repo"


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    workspace = tmp_path / "toy_repo"
    shutil.copytree(
        TOY_REPO, workspace, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    return workspace


def _run_pytest(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_toy_repo_has_exactly_one_failing_test(toy_repo: Path) -> None:
    result = _run_pytest(toy_repo)

    assert result.returncode == 1, result.stdout
    assert "1 failed, 4 passed" in result.stdout
    assert "test_median_of_even_length_list_averages_middle_pair" in result.stdout


def test_toy_repo_passes_after_reference_fix(toy_repo: Path) -> None:
    stats = toy_repo / "stats.py"
    source = stats.read_text()
    buggy = "    return ordered[middle]\n"
    fixed = (
        "    if len(ordered) % 2 == 0:\n"
        "        return (ordered[middle - 1] + ordered[middle]) / 2\n"
        "    return ordered[middle]\n"
    )
    assert source.count(buggy) == 1
    stats.write_text(source.replace(buggy, fixed))

    result = _run_pytest(toy_repo)

    assert result.returncode == 0, result.stdout
