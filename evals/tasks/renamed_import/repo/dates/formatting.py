"""Format dates for people."""

from datetime import date


def format_date(day: date) -> str:
    """`day` as day, short month and year: "3 Nov 2026"."""
    return f"{day.day} {day:%b %Y}"
