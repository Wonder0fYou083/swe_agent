"""Invoice totals."""

from orders import Order


def total(order: Order) -> float:
    """What the customer pays for `order`."""
    return order.unit_price * order.quantity
