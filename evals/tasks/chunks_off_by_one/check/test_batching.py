import pytest

from batching import chunks


def test_chunks_of_two() -> None:
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_exact_multiple_has_no_short_chunk() -> None:
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_size_larger_than_list_gives_one_chunk() -> None:
    assert chunks([1, 2, 3], 5) == [[1, 2, 3]]


def test_size_one() -> None:
    assert chunks([1, 2, 3], 1) == [[1], [2], [3]]


def test_empty_list_gives_no_chunks() -> None:
    assert chunks([], 3) == []


def test_size_zero_raises() -> None:
    with pytest.raises(ValueError):
        chunks([1, 2], 0)
