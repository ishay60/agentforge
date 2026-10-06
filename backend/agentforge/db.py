"""Persistence: stdlib sqlite3, one connection, plain SQL.

ponytail: sqlite file, not Postgres. Upgrade path: swap `connect()` for psycopg + the DDL below
(JSON TEXT -> JSONB, TEXT ids -> UUID) when >1 API process needs the same data. Nothing else changes.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = os.environ.get("AGENTFORGE_DB", "data/agentforge.db")
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, cfg TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id), filename TEXT NOT NULL,
    source_type TEXT NOT NULL, status TEXT NOT NULL, chunk_count INTEGER, file_size_bytes INTEGER,
    content_hash TEXT NOT NULL, processed_at TEXT);
CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id), name TEXT NOT NULL,
    tool_type TEXT NOT NULL, cfg TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id), title TEXT,
    message_count INTEGER NOT NULL DEFAULT 0, total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0, total_cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), role TEXT NOT NULL,
    content TEXT NOT NULL, model_used TEXT, cost_usd REAL, latency_ms INTEGER, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY, agent_id TEXT NOT NULL, conversation_id TEXT, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, latency_ms INTEGER, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_usage_agent_time ON usage_log(agent_id, created_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    """One autocommit connection, schema applied on first use.

    ponytail: shared across FastAPI's threadpool via check_same_thread=False; Python's sqlite3 serializes
    calls. Move to a per-request connection (or Postgres pool) if write contention ever shows up.
    """
    global _conn
    if _conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
    return _conn


def q(sql: str, *params) -> list[dict]:
    return [dict(r) for r in connect().execute(sql, params).fetchall()]


def one(sql: str, *params) -> dict | None:
    rows = q(sql, *params)
    return rows[0] if rows else None


# --- agents ---------------------------------------------------------------------

def insert_agent(aid: str, cfg: dict) -> dict:
    connect().execute("INSERT INTO agents VALUES (?,?,?,?)", (aid, cfg["name"], json.dumps(cfg), now()))
    return get_agent(aid)


def get_agent(aid: str) -> dict | None:
    a = one("SELECT * FROM agents WHERE id=?", aid)
    return {**a, "cfg": json.loads(a["cfg"])} if a else None


def agent_summary(aid: str) -> dict:
    """The /agents list shape; counts via subqueries so the response is one round trip."""
    return one("""SELECT a.id, a.name, 'active' AS status,
            (SELECT COUNT(*) FROM documents WHERE agent_id=a.id) AS document_count,
            (SELECT COUNT(*) FROM tools WHERE agent_id=a.id) AS tool_count,
            (SELECT COUNT(*) FROM conversations WHERE agent_id=a.id) AS total_conversations,
            (SELECT ROUND(COALESCE(SUM(total_cost_usd),0),6) FROM conversations WHERE agent_id=a.id) AS total_cost_usd,
            a.created_at FROM agents a WHERE a.id=?""", aid)


def list_agents() -> list[dict]:
    return [agent_summary(r["id"]) for r in q("SELECT id FROM agents ORDER BY created_at")]


# --- documents / tools ------------------------------------------------------------

DOC_COLS = ("id", "filename", "source_type", "status", "chunk_count", "file_size_bytes", "content_hash", "processed_at")


def insert_document(aid: str, doc: dict) -> None:
    connect().execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                      (doc["id"], aid, *(doc[k] for k in DOC_COLS[1:])))


def list_documents(aid: str) -> list[dict]:
    return q(f"SELECT {','.join(DOC_COLS)} FROM documents WHERE agent_id=? ORDER BY processed_at", aid)


def has_document_hash(aid: str, h: str) -> bool:
    return one("SELECT 1 FROM documents WHERE agent_id=? AND content_hash=?", aid, h) is not None


def insert_tool(aid: str, cfg: dict) -> None:
    connect().execute("INSERT INTO tools VALUES (?,?,?,?,?)",
                      (cfg["id"], aid, cfg["name"], cfg["tool_type"], json.dumps(cfg)))


def list_tools(aid: str) -> list[dict]:
    return [json.loads(r["cfg"]) for r in q("SELECT cfg FROM tools WHERE agent_id=? ORDER BY rowid", aid)]


# --- conversations / messages / usage ----------------------------------------------

CONV_COLS = "id, title, message_count, total_cost_usd, created_at, updated_at"


def get_or_create_conversation(aid: str, cid: str, title: str) -> dict:
    connect().execute("INSERT OR IGNORE INTO conversations (id, agent_id, title, created_at, updated_at) "
                      "VALUES (?,?,?,?,?)", (cid, aid, title, now(), now()))
    return one("SELECT * FROM conversations WHERE id=?", cid)


def get_conversation(aid: str, cid: str) -> dict | None:
    return one("SELECT * FROM conversations WHERE id=? AND agent_id=?", cid, aid)


def list_conversations(aid: str) -> list[dict]:
    return q(f"SELECT {CONV_COLS} FROM conversations WHERE agent_id=? ORDER BY created_at", aid)


def record_turn(aid: str, cid: str, user_msg: str, assistant_msg: str, model: str | None,
                input_tokens: int, output_tokens: int, cost_usd: float, latency_ms: int) -> None:
    """Persist one finished chat turn. Token/cost args are per-turn deltas."""
    import uuid
    ts = now()
    c = connect()
    c.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
              (str(uuid.uuid4()), cid, "user", user_msg, None, None, None, ts))
    c.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)",
              (str(uuid.uuid4()), cid, "assistant", assistant_msg, model, cost_usd, latency_ms, ts))
    c.execute("INSERT INTO usage_log (agent_id, conversation_id, model, input_tokens, output_tokens, cost_usd, "
              "latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?)",
              (aid, cid, model, input_tokens, output_tokens, cost_usd, latency_ms, ts))
    c.execute("UPDATE conversations SET message_count=message_count+2, total_input_tokens=total_input_tokens+?, "
              "total_output_tokens=total_output_tokens+?, total_cost_usd=ROUND(total_cost_usd+?,6), updated_at=? "
              "WHERE id=?", (input_tokens, output_tokens, cost_usd, ts, cid))


def list_messages(cid: str) -> list[dict]:
    return q("SELECT id, role, content, model_used, cost_usd, latency_ms, created_at FROM messages "
             "WHERE conversation_id=? ORDER BY rowid", cid)


# --- analytics (spec 6.6) ------------------------------------------------------------

def analytics(aid: str, period: str) -> dict:
    """One usage_log row == one turn == one user + one assistant message, hence COUNT(*)*2.
    top_topics omitted: YAGNI until there is a topic extractor to feed it."""
    since = (datetime.now(timezone.utc) - timedelta(days=int(period.rstrip("d")))).isoformat(timespec="seconds")
    where, args = "WHERE agent_id=? AND created_at>=?", (aid, since)
    s = one(f"""SELECT COUNT(DISTINCT conversation_id) AS total_conversations, COUNT(*)*2 AS total_messages,
            ROUND(COALESCE(SUM(cost_usd),0),6) AS total_cost_usd, CAST(COALESCE(AVG(latency_ms),0) AS INTEGER) AS avg_latency_ms
            FROM usage_log {where}""", *args)
    s["avg_cost_per_conversation"] = round(s["total_cost_usd"] / s["total_conversations"], 6) if s["total_conversations"] else 0.0
    daily = q(f"""SELECT substr(created_at,1,10) AS date, COUNT(DISTINCT conversation_id) AS conversations,
              COUNT(*)*2 AS messages, ROUND(SUM(cost_usd),6) AS cost_usd FROM usage_log {where}
              GROUP BY date ORDER BY date""", *args)
    models = q(f"""SELECT COALESCE(model,'unknown') AS model, COUNT(*) AS calls, ROUND(SUM(cost_usd),6) AS cost_usd
               FROM usage_log {where} GROUP BY model ORDER BY cost_usd DESC""", *args)
    return {"period": period, "summary": s, "daily": daily, "model_breakdown": models}
