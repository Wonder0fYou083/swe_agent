"""Format temperature readings for display."""

from temperature import to_fahrenheit


def format_reading(celsius: float, unit: str = "C") -> str:
    """`celsius` shown in `unit`, "C" or "F", to one decimal place: "21.5 °C", "70.7 °F"."""
    if unit == "C":
        return f"{celsius:.1f} °C"
    if unit == "F":
        return f"{to_fahrenheit(celsius):.1f} °F"
    raise ValueError(f"unknown unit {unit!r}")
