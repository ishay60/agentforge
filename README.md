# AgentForge

Build AI agents from your documents. Spec: `~/Projects/claude-work/ai_agent_platform_spec.md`.

## Status

Day 1–2 done: ingestion (PDF / URL / txt / md / html → chunks) and hybrid retrieval (FAISS + BM25 + RRF) with per-agent on-disk indexes.

## Run

```bash
cd backend
uv run python -m agentforge.cli ingest demo https://example.com ./some.pdf
uv run python -m agentforge.cli query demo "how do I reset my password"
uv run pytest
```

Indexes live in `backend/data/indexes/<agent_id>/`.
