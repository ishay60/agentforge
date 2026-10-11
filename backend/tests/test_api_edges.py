"""Edge cases and management endpoints: empty/oversized/bad uploads, tool toggle, agent delete."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentforge import agent, api, db, retriever

ECHO = f"{sys.executable} {Path(__file__).parent / 'mcp_echo_server.py'}"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever, "INDEX_ROOT", tmp_path / "idx")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(db, "_conn", None)
    monkeypatch.setattr(agent, "CHECKPOINT_DB", str(tmp_path / "cp.db"))
    monkeypatch.setattr(agent, "_saver", None)
    api._graphs.clear()
    api._retrievers.clear()
    api._hits.clear()
    return TestClient(api.app)


def test_upload_edge_cases(client, monkeypatch):
    aid = client.post("/api/v1/agents", json={"name": "e"}).json()["id"]
    r = client.post(f"/api/v1/agents/{aid}/documents", files={"file": ("empty.txt", b"   \n", "text/plain")})
    assert r.status_code == 422 and "no extractable text" in r.text
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    r = client.post(f"/api/v1/agents/{aid}/documents", files={"file": ("big.txt", b"x" * 11, "text/plain")})
    assert r.status_code == 413
    r = client.post(f"/api/v1/agents/{aid}/documents", json={"url": "http://127.0.0.1:9/nothing"})
    assert r.status_code == 502
    r = client.post(f"/api/v1/agents/{aid}/documents", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422
    assert client.get(f"/api/v1/agents/{aid}/documents").json()["documents"] == []
    assert client.post(f"/api/v1/agents/{aid}/chat", json={"message": "x" * 20001}).status_code == 422
    assert client.post("/api/v1/agents/nope/chat", json={"message": "hi"}).status_code == 404


def test_tool_toggle_and_agent_delete(client):
    aid = client.post("/api/v1/agents", json={"name": "t"}).json()["id"]
    assert client.get(f"/api/v1/agents/{aid}").json()["cfg"]["top_k"] == 5
    tid = client.post(f"/api/v1/agents/{aid}/tools", json={
        "name": "ping", "description": "d", "http_url": "https://x/{a}",
        "parameters_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}).json()["id"]
    client.post(f"/api/v1/agents/{aid}/tools/mcp", json={"target": ECHO})
    names = lambda: [t.name for t in api._graph(aid).nodes["tools"].bound.tools_by_name.values()]  # noqa: E731
    assert {"ping", "echo", "fail"} <= set(names())

    r = client.patch(f"/api/v1/agents/{aid}/tools/{tid}", json={"is_enabled": False})
    assert r.status_code == 200 and r.json()["is_enabled"] is False
    assert "ping" not in names() and "echo" in names()  # graph rebuilt without the disabled tool
    assert client.patch(f"/api/v1/agents/{aid}/tools/zzz", json={"is_enabled": True}).status_code == 404

    assert client.delete(f"/api/v1/agents/{aid}/tools/{tid}").status_code == 204
    assert client.delete(f"/api/v1/agents/{aid}/tools/{tid}").status_code == 404
    assert len(client.get(f"/api/v1/agents/{aid}/tools").json()["tools"]) == 2

    client.post(f"/api/v1/agents/{aid}/documents", files={"file": ("a.txt", b"hello world " * 5, "text/plain")})
    assert (retriever.INDEX_ROOT / aid / "index.faiss").exists()
    assert client.delete(f"/api/v1/agents/{aid}").status_code == 204
    assert client.get(f"/api/v1/agents/{aid}").status_code == 404
    assert not (retriever.INDEX_ROOT / aid).exists() and not (api.UPLOAD_DIR / aid).exists()
    assert client.get("/api/v1/agents").json()["total"] == 0
