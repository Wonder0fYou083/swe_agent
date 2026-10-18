from importlib.metadata import version

from typer.testing import CliRunner

from swe_agent.cli.main import app


def test_version_flag_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"swe-agent {version('swe-agent')}"
