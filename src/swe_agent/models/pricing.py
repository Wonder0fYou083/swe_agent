"""Per-model prices in USD per million tokens."""

from pydantic import BaseModel

from swe_agent.models.types import Usage


class Price(BaseModel):
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float  # 5-minute TTL


# Check the pricing page before relying on these. Haiku 5.5 has a second, higher rate card
# for prompts over 100K tokens ($0.50 / $2.50); Phase 1 runs stay far below that.
PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        cache_read_per_mtok=0.01,  # 0.1x input
        cache_write_per_mtok=0.125,  # 1.25x input
    ),
}


class UnknownModelError(KeyError):
    pass


def price_for(model: str) -> Price:
    """Look up `model` in PRICES. Raise UnknownModelError if it is missing.

    Failing here, when the adapter is built, is the point: without a price the
    max-cost budget silently stops working.
    """
    if PRICES.get(model) is None:
        raise UnknownModelError
    return Price(
        input_per_mtok=PRICES[model].input_per_mtok,
        output_per_mtok=PRICES[model].output_per_mtok,
        cache_read_per_mtok=PRICES[model].cache_read_per_mtok,
        cache_write_per_mtok=PRICES[model].cache_write_per_mtok,
    )


def cost_usd(usage: Usage, price: Price) -> float:
    """Each of the four counters times its own price, summed, divided by 1,000,000."""
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_tokens * price.cache_read_per_mtok
        + usage.cache_write_tokens * price.cache_write_per_mtok
    ) / 1e6
