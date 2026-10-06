# Architecture

AgentForge is a monolith on purpose: one FastAPI process serves ingestion, chat, tools, and analytics. Every component is a plain Python module with functions; the two classes in the codebase (`HybridRetriever`, `JsonFormatter`) exist because they own state that genuinely belongs together.

## One request, every hop

`POST /api/v1/agents/{id}/chat` with `{"message": "...", "conversation_id": "..."}`:

1. **Rate limiter** (`api.py`): sliding window per client IP, in-process deque.
2. **Graph lookup**: compiled LangGraph per agent, cached in-process, rebuilt when documents or tools change.
3. **Checkpointer**: the conversation id is the LangGraph `thread_id`. Prior messages load from sqlite; nothing is re-sent by the client.
4. **route**: a fresh user turn with a non-empty index goes to `retrieve`; a tool follow-up goes straight to `generate`.
5. **retrieve**: hybrid search (below) returns the top-k chunks with scores. Emitted to the client as a `retrieval` SSE event before any token.
6. **generate**: system prompt, a transient context message built from the chunks, then the conversation. Called through `invoke_with_failover` with the tool list bound. Tokens stream as `token` events. The context message is never persisted into the thread, so the checkpoint stays small and the next turn gets fresh retrieval.
7. **tools** (only if the model requested any): LangGraph `ToolNode` runs calls in parallel. Exceptions become `ToolMessage`s with `handle_tool_errors=True`, so a failing HTTP endpoint or a rejected SQL query comes back to the model as text it can reason about. Each call is a `tool_call` and `tool_result` event.
8. Loop back to `generate` until the model answers without tool calls or hits the 8-round cap.
9. **finalize**: cost from the cumulative token counts and the model actually used.
10. **done** event with tokens, cost, model, latency. The API layer stores the per-turn delta in `usage_log` and both messages in `messages`, and logs one JSON line.

## Retrieval

Dense search alone misses exact tokens (error codes, SKUs, names); BM25 alone misses paraphrase. Both run over the same chunk list and are merged with Reciprocal Rank Fusion, `score = Σ weight / (60 + rank)`, with `alpha` weighting dense vs sparse per agent. Over-fetching three times `top_k` from each side before fusion keeps results that rank well on only one side.

Embeddings come from `all-MiniLM-L6-v2` locally: 384 dimensions, fast on CPU, no per-document API cost, and the model is baked into the Docker image so cold start does not download anything.

Indexes are one FAISS flat index plus one `chunks.json` per agent on disk. Flat inner product is exact and fine to roughly a million vectors; the upgrade is an IVF or HNSW index behind the same `search()` signature. BM25 is rebuilt over the whole corpus on each add, which is O(n) and acceptable to about 100k chunks.

Chunks are 512 characters with 50 overlap from a recursive splitter. Characters, not tokens, so no tokenizer dependency at ingestion time; the ratio is close enough for retrieval quality.

## LLM layer

`llm.py` is about 60 lines: a cache of client instances keyed by provider, model, and temperature, and a circuit breaker. Three failures within 60 seconds skip that provider for the rest of the window. The chain is primary then the other cloud provider. Cost is a table lookup per model with a conservative default for unknown models. The intent is that an OpenAI outage is a log line, not a user-facing error.

Token usage is read from LangChain's `usage_metadata`, which both providers populate consistently, including under streaming.

## Tools and MCP

A tool is a LangChain `StructuredTool`. Three sources feed the same list:

- **Built-ins**: `calculate` walks an AST and only permits arithmetic nodes, so there is no `eval`. `search_knowledge_base` is a closure over the agent's retriever so the model can look things up mid-reasoning after the initial retrieval.
- **HTTP tools**: a config dict (URL template, method, headers, JSON schema) becomes an async callable. Path placeholders are filled from arguments; the rest become query parameters or a JSON body.
- **MCP tools**: `discover_mcp` connects once and lists tools; each discovered tool becomes a `StructuredTool` whose schema is the server's own JSON schema. Transport is chosen by the target string: an `http(s)` URL uses streamable HTTP, anything else is a stdio command. A fresh session is opened per call, which costs a handshake but avoids holding sessions across requests. This was verified against [mcpolyglot](https://github.com/ishay60/mcpolyglot)'s SQLite server: three tools discovered, emails redacted in results, and a `DELETE` without `WHERE` rejected and surfaced to the model as a tool error.

Every HTTP and MCP result is wrapped in an `<untrusted-data>` block with an instruction to treat it as data. That is the mitigation for prompt injection through fetched content; the same pattern mcpolyglot applies server-side.

## Persistence

Stdlib `sqlite3`, plain SQL, one autocommit connection shared across FastAPI's threadpool. Tables mirror the spec's Postgres schema with JSON stored as text. The LangGraph checkpointer is a second sqlite file. The upgrade is `psycopg` plus `AsyncPostgresSaver` behind the same function signatures; nothing above `db.py` changes.

Conversation history is served from the checkpointer, not the `messages` table, because the checkpoint includes tool messages in order. The `messages` and `usage_log` tables are the analytics copy and the audit trail.

## Streaming

LangGraph's `astream_events` is mapped to a small SSE vocabulary: `metadata`, `retrieval`, `token`, `tool_call`, `tool_result`, `done`. The client is a `fetch` plus `ReadableStream` parser, because `EventSource` cannot send a POST body. nginx is configured with `proxy_buffering off` so tokens are not held back.

If the model produces no streamed tokens (a non-streaming provider, or the all-providers-down fallback message), the final assistant text is sent as a single `token` so the client never ends a turn with an empty bubble.

## Evaluation

`backend/tests/eval` has 19 cases in four categories: RAG accuracy, tool selection, out-of-scope refusal, and multi-step. In scripted mode a fake chat model replays scripted responses including tool calls, and the assertions check the harness: the expected source was retrieved, the expected tool ran with the expected arguments, the answer contains or avoids given phrases. In live mode the same cases run against a real model. The scripted mode is what CI runs, so a regression in retrieval or tool dispatch fails the build without an API key.

## Deliberate simplifications

Each is marked `ponytail:` in the code with its upgrade path.

| Simplification | Ceiling | Upgrade |
|---|---|---|
| sqlite, one connection | one process | Postgres, connection pool |
| in-memory rate limiter | one process | Redis `INCR` with TTL |
| inline ingestion | large PDFs block the request | background task plus status polling |
| BM25 rebuilt per add | ~100k chunks | incremental index or a search engine |
| flat FAISS index | ~1M vectors | IVF or HNSW |
| new MCP session per call | handshake per call | long-lived session in a task group |
| no auth, no tenancy | single operator | API keys, `tenant_id` on every table |

## What I would do with more time

Postgres with per-tenant row-level security, a job queue for ingestion, a reranker between retrieval and generation, semantic caching of repeated queries, and tracing of each graph run so cost and latency can be attributed to individual nodes.
