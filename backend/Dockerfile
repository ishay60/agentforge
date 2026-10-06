# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY agentforge ./agentforge
# Pre-download the embedding model so the first request doesn't pay for it.
ENV HF_HOME=/app/.hf
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

FROM python:3.12-slim
RUN useradd -m -u 1000 app
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH HF_HOME=/app/.hf HF_HUB_OFFLINE=1
COPY --from=builder --chown=app:app /app /app
RUN mkdir -p data && chown app:app data
USER app
EXPOSE 8000
CMD ["uvicorn", "agentforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
