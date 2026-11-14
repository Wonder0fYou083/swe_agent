import pytest

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


def test_remove_lowers_the_count() -> None:
    stock = Inventory()
    stock.add("apple", 3)
    stock.remove("apple", 2)
    assert stock.count("apple") == 1


def test_remove_defaults_to_one() -> None:
    stock = Inventory()
    stock.add("apple", 3)
    stock.remove("apple")
    assert stock.count("apple") == 2


def test_an_item_at_zero_is_gone_from_items() -> None:
    stock = Inventory()
    stock.add("apple", 3)
    stock.add("pear")
    stock.remove("pear")
    assert stock.items() == [("apple", 3)]
    assert stock.count("pear") == 0


def test_removing_more_than_in_stock_raises_and_changes_nothing() -> None:
    stock = Inventory()
    stock.add("apple", 2)
    with pytest.raises(ValueError):
        stock.remove("apple", 3)
    assert stock.count("apple") == 2


def test_removing_an_item_never_added_raises() -> None:
    stock = Inventory()
    with pytest.raises(ValueError):
        stock.remove("plum")
    assert stock.items() == []


def test_removing_less_than_one_raises() -> None:
    stock = Inventory()
    stock.add("apple", 2)
    with pytest.raises(ValueError):
        stock.remove("apple", 0)
    assert stock.count("apple") == 2
