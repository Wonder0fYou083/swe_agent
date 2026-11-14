import pytest
from temperature import to_fahrenheit, to_kelvin


def test_to_fahrenheit() -> None:
    assert to_fahrenheit(100) == pytest.approx(212.0)


def test_to_kelvin() -> None:
    assert to_kelvin(0) == pytest.approx(273.15)
    assert to_kelvin(100) == pytest.approx(373.15)
    assert to_kelvin(-273.15) == pytest.approx(0.0)
