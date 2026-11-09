import pytest

from swe_agent.models.pricing import Price, UnknownModelError, cost_usd, price_for
from swe_agent.models.types import Usage


def test_cost_prices_each_counter_separately() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    price = Price(
        input_per_mtok=1, output_per_mtok=2, cache_read_per_mtok=3, cache_write_per_mtok=4
    )

    assert cost_usd(usage, price) == pytest.approx(10.0)


def test_unknown_model_fails_fast() -> None:
    with pytest.raises(UnknownModelError):
        price_for("no-such-model")
