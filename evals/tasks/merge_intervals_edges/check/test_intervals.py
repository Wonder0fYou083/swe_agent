from intervals import merge


def test_overlapping_pair_merges() -> None:
    assert merge([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]


def test_unsorted_input() -> None:
    assert merge([(8, 10), (1, 3), (2, 6)]) == [(1, 6), (8, 10)]


def test_touching_intervals_merge() -> None:
    assert merge([(1, 3), (3, 5)]) == [(1, 5)]


def test_contained_interval_disappears() -> None:
    assert merge([(1, 10), (2, 3)]) == [(1, 10)]


def test_empty_input() -> None:
    assert merge([]) == []


def test_disjoint_intervals_stay_apart() -> None:
    assert merge([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]
