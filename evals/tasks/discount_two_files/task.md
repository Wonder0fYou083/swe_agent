Orders need an optional discount.

- Add a field `discount_percent: float = 0.0` to `Order` in orders.py. Creating an `Order`
  with a discount below 0 or above 100 raises `ValueError`.
- `total` in invoice.py applies the discount: 2 items at 10.0 with `discount_percent=25`
  cost 15.0.
