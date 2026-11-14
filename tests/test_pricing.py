import pytest

from swe_agent.cli.main import Provider, build_model
from swe_agent.models.anthropic import AnthropicModel
from swe_agent.models.pricing import PRICES, Price, UnknownModelError, cost_usd, price_for
from swe_agent.models.types import Usage

CLAUDE_MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-fable-5-2",
]


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_every_claude_model_has_a_price_and_builds_an_adapter(model: str) -> None:
    assert price_for(model) == PRICES[model]
    llm = build_model(Provider.anthropic, model, "test-key", base_url=None, cache=True)

    assert isinstance(llm, AnthropicModel)
    assert llm.name == model


@pytest.mark.parametrize("model", [m for m in PRICES if m.startswith("claude-")])
def test_claude_cache_prices_follow_the_rate_card(model: str) -> None:
    price = PRICES[model]

    assert price.cache_write_per_mtok == pytest.approx(1.25 * price.input_per_mtok)
    assert 0 < price.cache_read_per_mtok < price.input_per_mtok
    assert price.output_per_mtok == pytest.approx(5 * price.input_per_mtok)


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
