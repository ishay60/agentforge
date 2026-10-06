"""FastAPI app. Run: uv run uvicorn agentforge.api:app --reload

ponytail: all state is in-process dicts + on-disk FAISS indexes. Swap AGENTS for
Postgres (spec section 5) and the rate limiter for Redis INCR before running >1 worker.
"""

import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentforge import agent as agent_mod
from agentforge.ingestion import ingest
from agentforge.retriever import HybridRetriever
from agentforge.tools import build_tools, discover_mcp

app = FastAPI(title="AgentForge", version="0.1.0")
API = "/api/v1"
UPLOAD_DIR = Path("data/uploads")

AGENTS: dict[str, dict] = {}       # id -> {cfg, documents, tools, conversations}
_graphs: dict[str, object] = {}    # id -> compiled graph (rebuilt when docs/tools change)
_retrievers: dict[str, HybridRetriever] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _agent(agent_id: str) -> dict:
    if agent_id not in AGENTS:
        raise HTTPException(404, "agent not found")
    return AGENTS[agent_id]


def _retriever(agent_id: str) -> HybridRetriever:
    if agent_id not in _retrievers:
        _retrievers[agent_id] = HybridRetriever.load(agent_id)
    return _retrievers[agent_id]


def _graph(agent_id: str):
    if agent_id not in _graphs:
        a = _agent(agent_id)
        r = _retriever(agent_id)
        _graphs[agent_id] = agent_mod.build_graph(a["cfg"], r, build_tools(r, a["tools"], a["cfg"]["top_k"]))
    return _graphs[agent_id]


# --- rate limiting ------------------------------------------------------------

RATE_LIMIT, RATE_WINDOW = 60, 60.0
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Sliding window per client IP. ponytail: per-process dict; Redis INCR for multi-worker."""
    key = request.client.host if request.client else "anon"
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return StreamingResponse(iter([b'{"detail":"rate limit exceeded"}']), status_code=429,
                                 media_type="application/json")
    q.append(now)
    resp = await call_next(request)
    resp.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT - len(q))
    return resp


# --- agents --------------------------------------------------------------------

class AgentIn(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = "You are a helpful assistant. Answer from the provided documentation; if unsure, say so."
    llm_provider: str = Field("anthropic", pattern="^(anthropic|openai)$")
    llm_model: str = "claude-sonnet-4-5"
    temperature: float = Field(0.3, ge=0, le=2)
    chunk_size: int = Field(512, ge=100, le=4000)
    top_k: int = Field(5, ge=1, le=20)
    hybrid_alpha: float = Field(0.7, ge=0, le=1)


def _agent_out(aid: str, a: dict) -> dict:
    return {"id": aid, "name": a["cfg"]["name"], "status": "active", "document_count": len(a["documents"]),
            "tool_count": len(a["tools"]), "total_conversations": len(a["conversations"]),
            "total_cost_usd": round(sum(c["total_cost_usd"] for c in a["conversations"].values()), 6),
            "created_at": a["created_at"]}


@app.post(f"{API}/agents", status_code=201)
def create_agent(body: AgentIn):
    aid = str(uuid.uuid4())
    AGENTS[aid] = {"cfg": body.model_dump(), "documents": [], "tools": [], "conversations": {}, "created_at": _now()}
    return _agent_out(aid, AGENTS[aid])


@app.get(f"{API}/agents")
def list_agents():
    return {"agents": [_agent_out(k, v) for k, v in AGENTS.items()], "total": len(AGENTS)}


# --- documents -----------------------------------------------------------------

def _index(aid: str, source: str, filename: str, source_type: str, size: int) -> dict:
    a = _agent(aid)
    h, chunks = ingest(source, chunk_size=a["cfg"]["chunk_size"])
    if any(d["content_hash"] == h for d in a["documents"]):
        raise HTTPException(409, "identical document already indexed")
    r = _retriever(aid)
    r.add(chunks)
    r.save(aid)
    _graphs.pop(aid, None)  # tool list depends on index being non-empty
    doc = {"id": str(uuid.uuid4()), "filename": filename, "source_type": source_type, "status": "ready",
           "chunk_count": len(chunks), "file_size_bytes": size, "content_hash": h, "processed_at": _now()}
    a["documents"].append(doc)
    return doc


# ponytail: ingestion runs inline (seconds for a PDF). Returns 201 ready, not 202 processing.
# Add a background task + status polling when uploads get large.
@app.post(f"{API}/agents/{{agent_id}}/documents", status_code=201)
async def upload_document(agent_id: str, request: Request):
    """multipart/form-data with `file`, or JSON {"url": ...}."""
    _agent(agent_id)
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        url = (await request.json()).get("url")
        if not url:
            raise HTTPException(422, "url required")
        return _index(agent_id, url, url, "url", 0)
    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(422, 'send multipart `file` or JSON {"url": ...}')
    dest = UPLOAD_DIR / agent_id / Path(file.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    dest.write_bytes(data)
    return _index(agent_id, str(dest), file.filename, "upload", len(data))


@app.get(f"{API}/agents/{{agent_id}}/documents")
def list_documents(agent_id: str):
    return {"documents": _agent(agent_id)["documents"]}


# --- tools -----------------------------------------------------------------------

class HttpToolIn(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str
    http_url: str
    http_method: str = "GET"
    http_headers: dict[str, str] = {}
    parameters_schema: dict = {"type": "object", "properties": {}}


@app.post(f"{API}/agents/{{agent_id}}/tools", status_code=201)
def register_tool(agent_id: str, body: HttpToolIn):
    a = _agent(agent_id)
    cfg = {"id": str(uuid.uuid4()), "tool_type": "http", "is_enabled": True, **body.model_dump()}
    a["tools"].append(cfg)
    _graphs.pop(agent_id, None)
    return {k: cfg[k] for k in ("id", "name", "tool_type", "is_enabled")}


class McpIn(BaseModel):
    target: str = Field(description="http(s) URL of a streamable-HTTP MCP server, or a stdio command line")
    name: str = "mcp"


@app.post(f"{API}/agents/{{agent_id}}/tools/mcp", status_code=201)
async def connect_mcp(agent_id: str, body: McpIn):
    a = _agent(agent_id)
    try:
        specs = await discover_mcp(body.target)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach MCP server: {e}") from e
    sid = str(uuid.uuid4())
    for s in specs:
        a["tools"].append({"id": str(uuid.uuid4()), "tool_type": "mcp", "is_enabled": True, "mcp_server_id": sid,
                           "target": body.target, **s})
    _graphs.pop(agent_id, None)
    return {"mcp_server_id": sid, "tools_discovered": [{"name": s["name"], "description": s["description"]} for s in specs]}


@app.get(f"{API}/agents/{{agent_id}}/tools")
def list_tools(agent_id: str):
    return {"tools": [{k: t[k] for k in ("id", "name", "tool_type", "description", "is_enabled")}
                      for t in _agent(agent_id)["tools"]]}


# --- chat ----------------------------------------------------------------------------

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = None


@app.post(f"{API}/agents/{{agent_id}}/chat")
async def chat(agent_id: str, body: ChatIn):
    a = _agent(agent_id)
    graph = _graph(agent_id)
    cid = body.conversation_id or str(uuid.uuid4())
    conv = a["conversations"].setdefault(cid, {"id": cid, "title": body.message[:60], "message_count": 0,
                                               "total_cost_usd": 0.0, "created_at": _now(), "updated_at": _now()})

    async def stream():
        yield f"event: metadata\ndata: {json.dumps({'conversation_id': cid})}\n\n"
        t0 = time.monotonic()
        async for event, data in agent_mod.chat(graph, cid, body.message):
            if event == "done":
                data["latency_ms"] = int((time.monotonic() - t0) * 1000)
                conv["message_count"] += 2
                conv["total_cost_usd"] = round(data["cost_usd"], 6)  # cost_usd is cumulative per thread
                conv["updated_at"] = _now()
            yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get(f"{API}/agents/{{agent_id}}/conversations")
def list_conversations(agent_id: str):
    return {"conversations": list(_agent(agent_id)["conversations"].values())}


@app.get(f"{API}/agents/{{agent_id}}/conversations/{{conv_id}}/messages")
def conversation_messages(agent_id: str, conv_id: str):
    if conv_id not in _agent(agent_id)["conversations"]:
        raise HTTPException(404, "conversation not found")
    return {"messages": agent_mod.history(_graph(agent_id), conv_id)}
