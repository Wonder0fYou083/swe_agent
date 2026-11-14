from invoice import total
from orders import Order


def test_total_without_discount() -> None:
    assert total(Order(unit_price=10.0, quantity=2)) == 20.0
