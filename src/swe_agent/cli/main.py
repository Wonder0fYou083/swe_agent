"""The `swe-agent` command. Subcommands such as `run` get added here as phases land."""

from typing import Annotated

import typer

from swe_agent import __version__

app = typer.Typer(
    help="A model-agnostic, terminal-first software-engineering agent.",
    no_args_is_help=True,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"swe-agent {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """A model-agnostic, terminal-first software-engineering agent."""
