"""Document ingestion: source -> clean text -> chunks.

A chunk is a plain dict: {"content", "chunk_index", "page", "source"}.
"""

import hashlib
from pathlib import Path

import pymupdf
import httpx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)], 1-based pages, empty pages dropped."""
    with pymupdf.open(path) as doc:
        return [(i + 1, p.get_text()) for i, p in enumerate(doc) if p.get_text().strip()]


def load_url(url: str) -> str:
    resp = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "agentforge/0.1"})
    resp.raise_for_status()
    return html_to_text(resp.text)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    lines = (line.strip() for line in body.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def load_file(path: str | Path) -> list[tuple[int | None, str]]:
    """Dispatch on extension. Returns [(page_or_None, text)]."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        text = html_to_text(text)
    return [(None, text)]  # .txt / .md / anything else: passthrough


def content_hash(pages: list[tuple[int | None, str]]) -> str:
    h = hashlib.sha256()
    for _, text in pages:
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def chunk_pages(
    pages: list[tuple[int | None, str]],
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    # ponytail: chunk_size is characters, not tokens. ~4 chars/token; switch to
    # a token-based length_function if retrieval quality needs it.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"content": piece, "chunk_index": len(chunks), "page": page, "source": source})
    return chunks


def ingest(source: str, **chunk_kwargs) -> tuple[str, list[dict]]:
    """URL or file path -> (content_hash, chunks)."""
    if source.startswith(("http://", "https://")):
        pages = [(None, load_url(source))]
    else:
        pages = load_file(source)
    return content_hash(pages), chunk_pages(pages, source=source, **chunk_kwargs)
