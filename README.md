# AgentForge

Build AI agents from your documents. Spec: `~/Projects/claude-work/ai_agent_platform_spec.md`.

## Status

- Day 1–2: ingestion (PDF / URL / txt / md / html → chunks), hybrid retrieval (FAISS + BM25 + RRF), per-agent on-disk indexes.
- Day 3–4: LangGraph agent (`route → retrieve → generate ⇄ tools → finalize`), conversation memory via checkpointer, LLM failover with circuit breaker, cost tracking, tool registry (built-ins, HTTP webhooks, MCP servers over stdio or streamable HTTP).
- Day 5: FastAPI with SSE chat streaming, document upload, tools, conversation history, rate limiting.

## Run

```bash
cd backend
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY (used as fallback either way)

uv run python -m agentforge.cli ingest demo https://example.com ./some.pdf
uv run python -m agentforge.cli query demo "how do I reset my password"
uv run python -m agentforge.cli chat demo

uv run uvicorn agentforge.api:app --reload   # Swagger at http://localhost:8000/docs
uv run pytest                                # no API key needed; LLM is faked
```

## API (`/api/v1`)

| Method | Path | Notes |
|---|---|---|
| POST | `/agents` | create (name, system_prompt, llm_provider, llm_model, temperature, top_k, …) |
| GET | `/agents` | list with doc/tool/conversation counts and cost |
| POST | `/agents/{id}/documents` | multipart `file` or JSON `{"url": …}`; indexed inline, 409 on duplicate content |
| GET | `/agents/{id}/documents` | |
| POST | `/agents/{id}/tools` | HTTP webhook tool: `{name, description, http_url, http_method, parameters_schema}` |
| POST | `/agents/{id}/tools/mcp` | `{"target": "http://host/mcp"}` or `{"target": "node server.js serve"}`; auto-discovers tools |
| GET | `/agents/{id}/tools` | |
| POST | `/agents/{id}/chat` | SSE: `metadata`, `retrieval`, `token`, `tool_call`, `tool_result`, `done` |
| GET | `/agents/{id}/conversations` | |
| GET | `/agents/{id}/conversations/{cid}/messages` | full thread incl. tool messages |

```bash
curl -N localhost:8000/api/v1/agents/$ID/chat -H 'content-type: application/json' \
  -d '{"message":"what does the return policy say? also what is 17*23"}'
```

## How the agent decides RAG vs tools

Every fresh user turn runs hybrid retrieval first (when the agent has an index) and injects the top chunks as context. The model also gets `search_knowledge_base` as a tool for follow-up lookups mid-reasoning, plus `calculate`, `get_current_time`, and any registered HTTP/MCP tools. Tool exceptions become tool messages, so the model sees the error and recovers instead of the turn crashing. Tool loops are capped at 8 rounds. If every LLM provider fails, the turn returns an apology message rather than a 500.

Tool and MCP results are wrapped as `<untrusted-data>` (pattern borrowed from [mcpolyglot](https://github.com/ishay60/mcpolyglot)); verified end-to-end against mcpolyglot's SQLite server.

## Layout

```
backend/agentforge/
  ingestion.py   sources -> chunks
  retriever.py   HybridRetriever (FAISS + BM25 + RRF), save/load per agent
  llm.py         get_llm, invoke_with_failover, calculate_cost
  tools.py       built-ins, make_http_tool, discover_mcp / make_mcp_tool, build_tools
  agent.py       LangGraph graph, chat() SSE event generator, history()
  api.py         FastAPI app + rate limiter
  cli.py         ingest / query / chat
```

Known simplifications (each marked `ponytail:` in code): in-process agent/conversation state and checkpointer (Postgres next), per-process rate limiter (Redis next), inline ingestion (background task next), fresh MCP session per tool call.
