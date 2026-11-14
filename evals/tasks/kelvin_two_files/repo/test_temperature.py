import pytest
from temperature import to_fahrenheit


def test_to_fahrenheit() -> None:
    assert to_fahrenheit(100) == pytest.approx(212.0)
