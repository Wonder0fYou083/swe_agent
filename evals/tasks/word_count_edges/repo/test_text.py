from text import word_count


def test_simple_sentence() -> None:
    assert word_count("the quick brown fox") == 4
