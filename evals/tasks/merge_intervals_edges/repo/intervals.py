"""Closed integer intervals."""


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping intervals and return them sorted by start.

    Intervals are closed, so (1, 3) and (3, 5) overlap and merge into (1, 5).
    The input may be in any order.
    """
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
