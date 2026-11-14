"""Text helpers."""


def word_count(text: str) -> int:
    """The number of words in `text`. Words are separated by runs of whitespace."""
    return len(text.split(" "))
