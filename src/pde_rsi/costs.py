"""Model-call cost from provider usage records and configs/model_prices.json (USD per 1M tokens).

Shared by the harness spending cap, the native coding-agent driver, and scripts/api_cost.py so that every
configuration is charged by the same rule. Unknown prices are None; callers decide whether that is allowed.
"""

import json
from pathlib import Path

PRICES = Path(__file__).resolve().parents[2]/"configs/model_prices.json"


def load_prices(path=PRICES):
    return json.loads(Path(path).read_text())


def usage_tokens(usage):
    """(uncached input, cache read, cache write, output) from Bedrock Converse, Anthropic, or OpenAI usage."""
    if "inputTokens" in usage:  # Bedrock Converse
        return (usage.get("inputTokens") or 0, usage.get("cacheReadInputTokens") or 0,
                usage.get("cacheWriteInputTokens") or 0, usage.get("outputTokens") or 0)
    if "input_tokens" in usage:  # Anthropic messages / OpenAI Responses
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        read = usage.get("cache_read_input_tokens") or 0
        return ((usage.get("input_tokens") or 0)-cached, read+cached, usage.get("cache_creation_input_tokens") or 0,
                usage.get("output_tokens") or 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0  # OpenAI chat completions
    return ((usage.get("prompt_tokens") or 0)-cached, cached, 0, usage.get("completion_tokens") or 0)


def cost(tokens, price):
    """USD for (input, cache_read, cache_write, output) tokens, or None when a needed price is unknown.
    A cache class with no listed price is billed as ordinary input."""
    total = 0.0
    for count, key in zip(tokens, ("input", "cache_read", "cache_write", "output")):
        if count:
            rate = price.get(key)
            if rate is None and key in ("cache_read", "cache_write"):
                rate = price.get("input")
            if rate is None:
                return None
            total += count*rate/1e6
    return total


def response_cost(response_path, model, prices=None):
    """Cost of one saved provider response (model-call-*/response.json); 0 when no usage was returned."""
    prices = prices or load_prices()
    path = Path(response_path)
    if not path.exists():
        return 0.0
    usage = json.loads(path.read_text()).get("usage") or {}
    return cost(usage_tokens(usage), prices.get(model, {}))
