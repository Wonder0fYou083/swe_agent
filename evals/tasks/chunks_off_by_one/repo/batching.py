"""Split lists into batches."""


def chunks(items: list[int], size: int) -> list[list[int]]:
    """Split `items` into lists of `size` items, in order. The last list may be shorter."""
    if size < 1:
        raise ValueError("size must be at least 1")
    return [items[i : i + size] for i in range(0, len(items) - size, size)]
