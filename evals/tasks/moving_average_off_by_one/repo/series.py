"""Helpers for number series."""


def moving_average(values: list[float], k: int) -> list[float]:
    """The mean of each run of `k` consecutive values, in order.

    A list of n values has n - k + 1 such runs. Raises ValueError unless 1 <= k <= n.
    """
    if not 1 <= k <= len(values):
        raise ValueError("k must be between 1 and len(values)")
    return [sum(values[i : i + k]) / k for i in range(len(values) - k)]
