from text import word_count


def test_simple_sentence() -> None:
    assert word_count("the quick brown fox") == 4


def test_empty_text_has_no_words() -> None:
    assert word_count("") == 0


def test_only_spaces_has_no_words() -> None:
    assert word_count("   ") == 0


def test_double_spaces() -> None:
    assert word_count("one  two") == 2


def test_tabs_and_newlines() -> None:
    assert word_count("one\ttwo\nthree") == 3


def test_leading_and_trailing_spaces() -> None:
    assert word_count("  one two three ") == 3
