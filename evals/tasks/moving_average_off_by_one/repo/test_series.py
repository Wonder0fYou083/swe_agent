from series import moving_average


def test_windows_of_two() -> None:
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
