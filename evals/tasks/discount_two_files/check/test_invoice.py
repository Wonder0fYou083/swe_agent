import pytest

from invoice import total
from orders import Order


def test_total_without_discount() -> None:
    assert total(Order(unit_price=10.0, quantity=2)) == 20.0


def test_discount_is_applied() -> None:
    order = Order(unit_price=10.0, quantity=2, discount_percent=25)
    assert total(order) == pytest.approx(15.0)


def test_full_discount_is_free() -> None:
    assert total(Order(unit_price=10.0, quantity=2, discount_percent=100)) == pytest.approx(0.0)


def test_default_discount_is_zero() -> None:
    assert Order(unit_price=1.0, quantity=1).discount_percent == 0.0


def test_negative_discount_raises() -> None:
    with pytest.raises(ValueError):
        Order(unit_price=10.0, quantity=2, discount_percent=-1)


def test_discount_above_100_raises() -> None:
    with pytest.raises(ValueError):
        Order(unit_price=10.0, quantity=2, discount_percent=101)
