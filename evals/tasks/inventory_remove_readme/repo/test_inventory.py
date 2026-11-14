from inventory import Inventory


def test_add_and_count() -> None:
    stock = Inventory()
    stock.add("apple", 3)
    stock.add("apple")
    assert stock.count("apple") == 4
    assert stock.count("pear") == 0


def test_items_are_sorted_by_name() -> None:
    stock = Inventory()
    stock.add("pear")
    stock.add("apple", 2)
    assert stock.items() == [("apple", 2), ("pear", 1)]
