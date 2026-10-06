"""Agent loop + API with a scripted fake LLM: RAG retrieval -> tool call -> tool error recovery -> answer."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agentforge import api, llm, retriever
from agentforge.tools import calculate

BOUND: list[str] = []  # tool names the fake saw on bind_tools


class FakeLLM(BaseChatModel):
    """Scripted responses; streams each as a single chunk (incl. tool calls) like a real provider."""
    script: list

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kw):
        BOUND[:] = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        m = self.script.pop(0)
        chunks = [{"name": t["name"], "args": json.dumps(t["args"]), "id": t["id"], "index": i}
                  for i, t in enumerate(m.tool_calls)]
        yield ChatGenerationChunk(message=AIMessageChunk(content=m.content, tool_call_chunks=chunks,
                                                         usage_metadata=m.usage_metadata))


def _ai(text="", tool_calls=()):
    return AIMessage(content=text, tool_calls=list(tool_calls),
                     usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def test_calculate_is_arithmetic_only():
    assert calculate("(3 + 4) * 2 ** 2") == "28"
    with pytest.raises(ValueError):
        calculate("__import__('os').system('id')")


def test_agent_loop_and_api(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever, "INDEX_ROOT", tmp_path / "idx")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "up")
    fake = FakeLLM(script=[
        _ai(tool_calls=[{"name": "calculate", "args": {"expression": "6 * 7"}, "id": "c1"},
                        {"name": "calculate", "args": {"expression": "1 /"}, "id": "c2"}]),  # 2nd one errors
        _ai("The answer is 42; the second expression was invalid. Password reset: see [Source 1]."),
        _ai("Follow-up answered with memory."),
    ])
    monkeypatch.setattr(llm, "get_llm", lambda *a, **k: fake)

    c = TestClient(api.app)
    aid = c.post("/api/v1/agents", json={"name": "bot"}).json()["id"]
    assert c.get("/api/v1/agents").json()["total"] == 1

    doc = tmp_path / "pw.txt"
    doc.write_text("To reset your password click 'Forgot password' on the login page.")
    with doc.open("rb") as f:
        r = c.post(f"/api/v1/agents/{aid}/documents", files={"file": ("pw.txt", f, "text/plain")})
    assert r.status_code == 201 and r.json()["chunk_count"] == 1
    with doc.open("rb") as f:  # same content -> 409
        assert c.post(f"/api/v1/agents/{aid}/documents", files={"file": ("pw2.txt", f)}).status_code == 409

    r = c.post(f"/api/v1/agents/{aid}/chat", json={"message": "reset password? also 6*7"})
    events = [(l.split(": ", 1)[1], None) for l in r.text.splitlines() if l.startswith("event: ")]
    datas = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    kinds = [e for e, _ in events]
    assert kinds[0] == "metadata" and kinds[1] == "retrieval" and kinds[-1] == "done"
    assert kinds.count("tool_call") == 2 and kinds.count("tool_result") == 2
    assert "search_knowledge_base" in BOUND and "calculate" in BOUND
    assert datas[1]["chunks"][0]["source"].endswith("pw.txt")
    results = [d for e, d in zip(kinds, datas) if e == "tool_result"]
    texts = sorted(r["result"] for r in results)  # parallel tool calls finish in any order
    assert texts[0] == "42" and texts[1].startswith("error:")  # error surfaced as tool message, loop continued
    assert datas[-1]["input_tokens"] == 200 and datas[-1]["cost_usd"] > 0
    cid = datas[0]["conversation_id"]

    # memory: second turn in the same thread sees the whole history
    c.post(f"/api/v1/agents/{aid}/chat", json={"message": "and again?", "conversation_id": cid})
    msgs = c.get(f"/api/v1/agents/{aid}/conversations/{cid}/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool", "assistant", "user", "assistant"]
    assert msgs[-1]["content"] == "Follow-up answered with memory."
    convs = c.get(f"/api/v1/agents/{aid}/conversations").json()["conversations"]
    assert convs[0]["id"] == cid and convs[0]["message_count"] == 4

    # all providers down -> graceful assistant message, not a 500
    monkeypatch.setattr(llm, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    llm._failures.clear()
    r = c.post(f"/api/v1/agents/{aid}/chat", json={"message": "hi"})
    assert r.status_code == 200 and "couldn't reach" in r.text


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 2)
    api._hits.clear()
    c = TestClient(api.app)
    assert [c.get("/api/v1/agents").status_code for _ in range(3)] == [200, 200, 429]
    api._hits.clear()
