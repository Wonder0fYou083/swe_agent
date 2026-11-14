import pytest

from series import moving_average


def test_windows_of_two() -> None:
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_window_as_long_as_the_list_gives_its_mean() -> None:
    assert moving_average([2, 4, 6], 3) == [4.0]


def test_window_of_one_gives_the_values() -> None:
    assert moving_average([1, 5, 3], 1) == [1.0, 5.0, 3.0]


def test_window_of_zero_raises() -> None:
    with pytest.raises(ValueError):
        moving_average([1, 2], 0)


def test_window_longer_than_the_list_raises() -> None:
    with pytest.raises(ValueError):
        moving_average([1, 2], 3)
