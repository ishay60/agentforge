"""LLM provider layer: cached clients, circuit-breaker failover, cost arithmetic.

Module-level dicts, not a class. State is three dicts; a class adds ceremony.
"""

import logging
import time

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_instances: dict[str, object] = {}
_failures: dict[str, int] = {}
_last_fail: dict[str, float] = {}

_CONSTRUCTORS = {"anthropic": ChatAnthropic, "openai": ChatOpenAI}
_FALLBACKS = {"anthropic": ("openai", "gpt-4o"), "openai": ("anthropic", "claude-sonnet-4-5")}

# (input, output) USD per 1K tokens
COST_PER_1K = {
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = COST_PER_1K.get(model, (0.001, 0.003))
    return input_tokens / 1000 * inp + output_tokens / 1000 * out


def get_llm(provider: str, model: str, temperature: float = 0.3):
    key = f"{provider}:{model}:{temperature}"
    if key not in _instances:
        _instances[key] = _CONSTRUCTORS[provider](model=model, temperature=temperature)
    return _instances[key]


async def invoke_with_failover(provider: str, model: str, messages: list, tools=None, temperature: float = 0.3):
    """Try primary; fall back if it fails or is circuit-broken (3+ failures in 60s).

    Returns (response, provider_used, model_used).
    """
    candidates = [(provider, model)]
    if fb := _FALLBACKS.get(provider):
        candidates.append(fb)

    last_err: Exception | None = None
    for p, m in candidates:
        if _failures.get(p, 0) >= 3 and time.time() - _last_fail.get(p, 0) < 60:
            continue
        try:
            llm = get_llm(p, m, temperature)
            if tools:
                llm = llm.bind_tools(tools)
            result = await llm.ainvoke(messages)
            _failures[p] = 0
            return result, p, m
        except Exception as e:  # noqa: BLE001 - any provider error trips the breaker
            _failures[p] = _failures.get(p, 0) + 1
            _last_fail[p] = time.time()
            last_err = e
            log.warning("llm_fail provider=%s model=%s err=%s", p, m, e)

    raise last_err or RuntimeError("all providers circuit-broken")
