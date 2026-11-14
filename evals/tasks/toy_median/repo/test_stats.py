import pytest
from stats import mean, median


def test_mean_of_list() -> None:
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_of_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])


def test_median_of_odd_length_list_is_middle_value() -> None:
    assert median([3, 1, 2]) == 2


def test_median_of_even_length_list_averages_middle_pair() -> None:
    assert median([4, 1, 3, 2]) == 2.5


def test_median_of_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        median([])
