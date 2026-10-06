"""FastAPI app. Run: uv run uvicorn agentforge.api:app --reload

ponytail: state lives in sqlite (agentforge/db.py) + on-disk FAISS indexes, graphs cached per process.
Swap the rate limiter for Redis INCR and sqlite for Postgres before running >1 worker.
"""

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentforge import agent as agent_mod
from agentforge import db
from agentforge.ingestion import ingest
from agentforge.retriever import HybridRetriever
from agentforge.tools import build_tools, discover_mcp

app = FastAPI(title="AgentForge", version="0.1.0")
# ponytail: open CORS; the nginx proxy makes this moot in Docker, it only matters for `vite dev` hitting :8000 directly
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
API = "/api/v1"
UPLOAD_DIR = Path("data/uploads")

_graphs: dict[str, object] = {}    # id -> compiled graph (rebuilt when docs/tools change)
_retrievers: dict[str, HybridRetriever] = {}


# --- logging: one JSON line per record --------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(r, "%Y-%m-%dT%H:%M:%S"), "level": r.levelname, "logger": r.name,
             "msg": r.getMessage()} | getattr(r, "data", {})
        if r.exc_info:
            d["exc"] = self.formatException(r.exc_info)
        return json.dumps(d, default=str)


if not logging.getLogger().handlers:  # configure once; uvicorn/pytest may already own the root
    _h = logging.StreamHandler()
    _h.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[_h])
log = logging.getLogger("agentforge")


def _agent(agent_id: str) -> dict:
    a = db.get_agent(agent_id)
    if a is None:
        raise HTTPException(404, "agent not found")
    return a


def _retriever(agent_id: str) -> HybridRetriever:
    if agent_id not in _retrievers:
        _retrievers[agent_id] = HybridRetriever.load(agent_id)
    return _retrievers[agent_id]


def _graph(agent_id: str):
    if agent_id not in _graphs:
        cfg, r = _agent(agent_id)["cfg"], _retriever(agent_id)
        _graphs[agent_id] = agent_mod.build_graph(cfg, r, build_tools(r, db.list_tools(agent_id), cfg["top_k"]))
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


@app.post(f"{API}/agents", status_code=201)
def create_agent(body: AgentIn):
    aid = str(uuid.uuid4())
    db.insert_agent(aid, body.model_dump())
    return db.agent_summary(aid)


@app.get(f"{API}/agents")
def list_agents():
    agents = db.list_agents()
    return {"agents": agents, "total": len(agents)}


# --- documents -----------------------------------------------------------------

def _index(aid: str, source: str, filename: str, source_type: str, size: int) -> dict:
    a = _agent(aid)
    h, chunks = ingest(source, chunk_size=a["cfg"]["chunk_size"])
    if db.has_document_hash(aid, h):
        raise HTTPException(409, "identical document already indexed")
    r = _retriever(aid)
    r.add(chunks)
    r.save(aid)
    _graphs.pop(aid, None)  # tool list depends on index being non-empty
    doc = {"id": str(uuid.uuid4()), "filename": filename, "source_type": source_type, "status": "ready",
           "chunk_count": len(chunks), "file_size_bytes": size, "content_hash": h, "processed_at": db.now()}
    db.insert_document(aid, doc)
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
    _agent(agent_id)
    return {"documents": db.list_documents(agent_id)}


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
    _agent(agent_id)
    cfg = {"id": str(uuid.uuid4()), "tool_type": "http", "is_enabled": True, **body.model_dump()}
    db.insert_tool(agent_id, cfg)
    _graphs.pop(agent_id, None)
    return {k: cfg[k] for k in ("id", "name", "tool_type", "is_enabled")}


class McpIn(BaseModel):
    target: str = Field(description="http(s) URL of a streamable-HTTP MCP server, or a stdio command line")
    name: str = "mcp"


@app.post(f"{API}/agents/{{agent_id}}/tools/mcp", status_code=201)
async def connect_mcp(agent_id: str, body: McpIn):
    _agent(agent_id)
    try:
        specs = await discover_mcp(body.target)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach MCP server: {e}") from e
    sid = str(uuid.uuid4())
    for s in specs:
        db.insert_tool(agent_id, {"id": str(uuid.uuid4()), "tool_type": "mcp", "is_enabled": True,
                                  "mcp_server_id": sid, "target": body.target, **s})
    _graphs.pop(agent_id, None)
    return {"mcp_server_id": sid, "tools_discovered": [{"name": s["name"], "description": s["description"]} for s in specs]}


@app.get(f"{API}/agents/{{agent_id}}/tools")
def list_tools(agent_id: str):
    return {"tools": [{k: t[k] for k in ("id", "name", "tool_type", "description", "is_enabled")}
                      for t in db.list_tools(_agent(agent_id)["id"])]}


# --- chat ----------------------------------------------------------------------------

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = None


@app.post(f"{API}/agents/{{agent_id}}/chat")
async def chat(agent_id: str, body: ChatIn):
    _agent(agent_id)
    graph = _graph(agent_id)
    cid = body.conversation_id or str(uuid.uuid4())
    conv = db.get_or_create_conversation(agent_id, cid, body.message[:60])

    async def stream():
        yield f"event: metadata\ndata: {json.dumps({'conversation_id': cid})}\n\n"
        t0, answer = time.monotonic(), []
        async for event, data in agent_mod.chat(graph, cid, body.message):
            if event == "token":
                answer.append(data["content"])
            elif event == "done":
                data["latency_ms"] = int((time.monotonic() - t0) * 1000)
                # state tokens/cost are cumulative per thread; store this turn's delta
                turn = {"model": data["model"], "input_tokens": data["input_tokens"] - conv["total_input_tokens"],
                        "output_tokens": data["output_tokens"] - conv["total_output_tokens"],
                        "cost_usd": round(data["cost_usd"] - conv["total_cost_usd"], 6), "latency_ms": data["latency_ms"]}
                db.record_turn(agent_id, cid, body.message, "".join(answer), **turn)
                log.info("chat_turn", extra={"data": {"agent_id": agent_id, "conversation_id": cid, **turn}})
            yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get(f"{API}/agents/{{agent_id}}/conversations")
def list_conversations(agent_id: str):
    _agent(agent_id)
    return {"conversations": db.list_conversations(agent_id)}


@app.get(f"{API}/agents/{{agent_id}}/conversations/{{conv_id}}/messages")
async def conversation_messages(agent_id: str, conv_id: str):
    if db.get_conversation(agent_id, conv_id) is None:
        raise HTTPException(404, "conversation not found")
    # ponytail: served from the LangGraph checkpoint (includes tool messages); the `messages` table is the
    # analytics/audit copy. Read from the table instead if checkpoints ever get pruned.
    return {"messages": await agent_mod.history(_graph(agent_id), conv_id)}


# --- analytics (spec 6.6) ---------------------------------------------------------------

@app.get(f"{API}/agents/{{agent_id}}/analytics")
def analytics(agent_id: str, period: str = "7d"):
    _agent(agent_id)
    if period not in ("7d", "30d"):
        raise HTTPException(422, "period must be 7d or 30d")
    return db.analytics(agent_id, period)
