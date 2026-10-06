"""Evaluation suite: RAG accuracy, tool selection, hallucination/refusal, multi-step.

Scripted (default):  uv run pytest tests/eval -q
Live model:          AGENTFORGE_EVAL_LIVE=1 ANTHROPIC_API_KEY=... uv run pytest tests/eval -q
"""

import uuid

import pytest

from agentforge import llm
from agentforge.agent import chat
from tests.eval.cases import EVAL_CASES
from tests.eval.conftest import LIVE, FakeLLM

REFUSAL_PHRASES = ("don't have information", "do not have information", "don't know", "cannot help",
                   "can't help", "not able to", "outside", "no information")


async def run_case(graph, query: str) -> dict:
    """Drive one turn and collapse chat() events into {response, retrieved_sources, tool_calls}."""
    out = {"response": "", "retrieved_sources": [], "tool_calls": []}
    async for event, data in chat(graph, str(uuid.uuid4()), query):
        if event == "retrieval":
            out["retrieved_sources"] = [c["source"] for c in data["chunks"]]
        elif event == "token":
            out["response"] += data["content"]
        elif event == "tool_call":
            out["tool_calls"].append({"name": data["name"], "args": data["args"]})
    return out


@pytest.mark.anyio
@pytest.mark.parametrize("case", EVAL_CASES, ids=[f"{c['category']}:{c['query'][:40]}" for c in EVAL_CASES])
async def test_eval_case(case, eval_graph, monkeypatch):
    if not LIVE:
        fake = FakeLLM(script=list(case["scripted"]))
        monkeypatch.setattr(llm, "get_llm", lambda *a, **k: fake)

    result = await run_case(eval_graph, case["query"])
    response = result["response"].lower()
    tools_used = [t for t in result["tool_calls"] if t["name"] != "search_knowledge_base"]

    if "expected_source" in case:
        assert case["expected_source"] in result["retrieved_sources"], result["retrieved_sources"]
    for phrase in case.get("expected_contains", []):
        assert phrase.lower() in response, f"missing {phrase!r} in {result['response']!r}"
    for phrase in case.get("must_not_contain", []):
        assert phrase.lower() not in response, f"hallucinated {phrase!r} in {result['response']!r}"
    if case.get("should_refuse"):
        assert any(p in response for p in REFUSAL_PHRASES), f"did not refuse: {result['response']!r}"
    if case.get("should_use_tool"):
        assert tools_used, "expected a tool call"
        if "expected_tool" in case:
            match = [t for t in tools_used if t["name"] == case["expected_tool"]]
            assert match, f"expected {case['expected_tool']}, got {[t['name'] for t in tools_used]}"
            expected_args = case.get("expected_tool_args", {})
            assert any(expected_args.items() <= (t["args"] or {}).items() for t in match), match
    elif "should_use_tool" in case:
        assert not tools_used, f"unexpected tool calls: {tools_used}"
