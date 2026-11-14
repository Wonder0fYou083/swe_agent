# inventory

```python
from inventory import Inventory

stock = Inventory()
stock.add("apple", 3)
stock.add("pear")
stock.count("apple")   # 3
stock.items()          # [("apple", 3), ("pear", 1)]

stock.remove("apple", 2)
stock.count("apple")   # 1
stock.remove("pear")   # qty defaults to 1
stock.items()          # [("apple", 1)]: an item whose count reaches 0 is gone
```

`remove(item, qty=1)` takes `qty` of `item` out of stock. It raises `ValueError` and changes
nothing when `qty` is less than 1 or more than is in stock, including for an item that was
never added.
