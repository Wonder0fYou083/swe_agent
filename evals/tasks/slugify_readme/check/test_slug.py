from slug import slugify


def test_readme_examples() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  Python 3.12  ") == "python-3-12"
    assert slugify("---") == ""


def test_uppercase_becomes_lowercase() -> None:
    assert slugify("ABC") == "abc"


def test_a_run_of_other_characters_becomes_one_dash() -> None:
    assert slugify("rock & roll") == "rock-roll"
    assert slugify("a__b") == "a-b"


def test_letters_and_digits_are_kept() -> None:
    assert slugify("x1y2") == "x1y2"


def test_empty_title() -> None:
    assert slugify("") == ""
