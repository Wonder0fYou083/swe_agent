import pytest
from report import format_reading


def test_celsius() -> None:
    assert format_reading(21.5) == "21.5 °C"


def test_fahrenheit() -> None:
    assert format_reading(21.5, "F") == "70.7 °F"


def test_unknown_unit_raises() -> None:
    with pytest.raises(ValueError):
        format_reading(21.5, "X")
