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
# Cache writes are 1.25x input on every Claude model; cache reads are not the same multiple.
PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        cache_read_per_mtok=0.01,  # 0.1x input
        cache_write_per_mtok=0.125,  # 1.25x input
    ),
    # Sonnet 5.5, Opus 5.5 and Fable 5.2: Anthropic API rates as of 2026-11-13.
    "claude-sonnet-5-5": Price(
        input_per_mtok=2.00,
        output_per_mtok=10.00,
        cache_read_per_mtok=0.10,  # 0.05x input
        cache_write_per_mtok=2.50,
    ),
    "claude-opus-5-5": Price(
        input_per_mtok=4.00,
        output_per_mtok=20.00,
        cache_read_per_mtok=0.20,  # 0.05x input
        cache_write_per_mtok=5.00,
    ),
    "claude-fable-5-2": Price(
        input_per_mtok=8.00,
        output_per_mtok=40.00,
        cache_read_per_mtok=0.20,  # 0.025x input
        cache_write_per_mtok=10.00,
    ),
    # OpenAI standard tier, up to 272K input tokens, from developers.openai.com/api/docs/pricing
    # on 2026-11-12. Caching is automatic and writing to the cache isn't billed.
    "gpt-6-luna": Price(
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        cache_read_per_mtok=0.01,  # 0.1x input
        cache_write_per_mtok=0.0,
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
