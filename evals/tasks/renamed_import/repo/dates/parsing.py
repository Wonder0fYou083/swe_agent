"""Parse ISO dates."""

from datetime import date


def parse_iso_date(text: str) -> date:
    """`text` as a date. It must look like 2026-11-13."""
    return date.fromisoformat(text)
