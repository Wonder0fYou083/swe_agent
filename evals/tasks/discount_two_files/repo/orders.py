"""Orders."""

from dataclasses import dataclass


@dataclass
class Order:
    unit_price: float
    quantity: int
