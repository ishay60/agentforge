"""Eval harness: in-memory corpus + retriever, a canned order-status tool, and the agent graph.

Scripted mode (default) swaps the LLM for a FakeLLM that replays each case's `scripted` messages.
Live mode (AGENTFORGE_EVAL_LIVE=1 + an API key) leaves agentforge.llm untouched.
"""

import json
import os

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool

from agentforge.agent import build_graph
from agentforge.ingestion import chunk_pages
from agentforge.retriever import HybridRetriever
from agentforge.tools import build_tools

LIVE = os.environ.get("AGENTFORGE_EVAL_LIVE") == "1" and bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a customer support assistant for Acme Store. Answer ONLY from the retrieved context or tool "
    "results; cite sources by number. If the context does not cover the question, reply exactly "
    "\"I don't have information about that.\" Use get_order_status for order lookups and calculate for math."
)

CORPUS = {
    "returns_policy.txt": (
        "Acme Store return policy. Items can be returned within 30 days of delivery for a full refund. "
        "A receipt or order confirmation email is required for all returns. Refunds are issued to the original "
        "payment method within 5 business days. Final-sale items and opened software cannot be returned."
    ),
    "password_reset.md": (
        "# Resetting your password\n\nClick 'Forgot password' on the login page and enter your email address. "
        "A reset link is sent to your inbox and expires after 24 hours. If you do not receive the email, "
        "check your spam folder or contact support."
    ),
    "shipping.txt": (
        "Shipping options. Standard shipping takes 3-5 business days and is free on orders over $50. "
        "Express shipping takes 1-2 business days and costs $15. We ship to the US and Canada only; "
        "international shipping is not available."
    ),
    "pricing.md": (
        "# Plans and pricing\n\nStarter plan: $10 per month, 1 user. Team plan: $25 per user per month, "
        "includes priority support. Enterprise plan: custom pricing, contact sales. Annual billing saves 20%."
    ),
}

ORDERS = {
    "ORD-99999": {"order_id": "ORD-99999", "status": "shipped", "carrier": "UPS", "eta": "2026-10-09"},
    "ORD-12345": {"order_id": "ORD-12345", "status": "processing", "eta": "2026-10-12"},
}


def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its id, e.g. ORD-12345."""
    return json.dumps(ORDERS.get(order_id, {"order_id": order_id, "error": "not found"}))


class FakeLLM(BaseChatModel):
    """Replays scripted AIMessages; streams each as one chunk (incl. tool calls) like a real provider."""
    script: list

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        m = self.script.pop(0)
        chunks = [{"name": t["name"], "args": json.dumps(t["args"]), "id": t["id"], "index": i}
                  for i, t in enumerate(m.tool_calls)]
        yield ChatGenerationChunk(message=AIMessageChunk(content=m.content, tool_call_chunks=chunks,
                                                         usage_metadata=m.usage_metadata))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def eval_retriever():
    r = HybridRetriever()
    for name, text in CORPUS.items():
        r.add(chunk_pages([(None, text)], source=name))
    return r


@pytest.fixture(scope="session")
def eval_graph(eval_retriever):
    provider = "openai" if LIVE and not os.environ.get("ANTHROPIC_API_KEY") else "anthropic"
    cfg = {"system_prompt": SYSTEM_PROMPT, "llm_provider": provider,
           "llm_model": "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-5",
           "temperature": 0, "top_k": 2, "hybrid_alpha": 0.7}
    tools = build_tools(eval_retriever, []) + [StructuredTool.from_function(get_order_status)]
    return build_graph(cfg, eval_retriever, tools)
