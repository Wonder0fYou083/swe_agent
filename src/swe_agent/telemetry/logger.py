from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import TypeAdapter

from swe_agent.telemetry.trace import SessionEnd, SessionStart, StepRecord, TraceRecord

# Validates one JSON line into whichever record its "type" field names.
_RECORD: TypeAdapter[TraceRecord] = TypeAdapter(TraceRecord)


class TrajectoryLogger:
    def __init__(self, path: Path) -> None:
        """Create the parent directory if needed and open `path` for appending (utf-8)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def write(self, record: SessionStart | StepRecord | SessionEnd) -> None:
        """Write record.model_dump_json() plus a newline, then flush.

        Flushing every line means a crash still leaves every step up to it on disk.
        """
        self._file.write(record.model_dump_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_trajectory(path: Path) -> list[TraceRecord]:
    """One record per non-empty line: _RECORD.validate_json(line)."""
    _file = path.open("r", encoding="utf-8")
    records: list[TraceRecord] = []
    for line in _file:
        if not line.strip():
            continue
        records.append(_RECORD.validate_json(line))
    _file.close()
    return records
