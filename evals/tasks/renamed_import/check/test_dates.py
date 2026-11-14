from datetime import date

from dates import format_date, parse_date


def test_parse_date() -> None:
    assert parse_date("2026-11-13") == date(2026, 11, 13)


def test_format_date() -> None:
    assert format_date(date(2026, 11, 3)) == "3 Nov 2026"


def test_round_trip() -> None:
    assert format_date(parse_date("2026-01-31")) == "31 Jan 2026"
