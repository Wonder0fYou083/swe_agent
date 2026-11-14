"""A small stock counter. See README.md."""


class Inventory:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def add(self, item: str, qty: int = 1) -> None:
        """Add `qty` of `item`. Raises ValueError when `qty` is less than 1."""
        if qty < 1:
            raise ValueError("qty must be at least 1")
        self._counts[item] = self._counts.get(item, 0) + qty

    def count(self, item: str) -> int:
        """How many of `item` are in stock; 0 for an item never added."""
        return self._counts.get(item, 0)

    def items(self) -> list[tuple[str, int]]:
        """Every item in stock with its count, sorted by name."""
        return sorted(self._counts.items())
