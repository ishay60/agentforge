"""Hybrid retrieval: FAISS dense + BM25 sparse, merged with Reciprocal Rank Fusion.

One instance per agent. State (encoder, index, corpus) belongs together -> class.
"""

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_ROOT = Path("data/indexes")


@lru_cache(maxsize=1)
def _encoder(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(name)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self):
        self.encoder = _encoder()
        self.index = faiss.IndexFlatIP(self.encoder.get_embedding_dimension())
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vecs = self.encoder.encode([c["content"] for c in chunks], normalize_embeddings=True)
        self.index.add(np.asarray(vecs, dtype=np.float32))
        self.chunks.extend(chunks)
        # ponytail: BM25 rebuilt over full corpus on every add. O(n) per add; fine
        # until corpora hit ~100k chunks, then move sparse search to a real engine.
        self.bm25 = BM25Okapi([_tokenize(c["content"]) for c in self.chunks])

    def search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> list[tuple[dict, float]]:
        """alpha weights dense results; (1 - alpha) weights sparse. RRF with k=60."""
        if not self.chunks:
            return []
        k_fetch = min(top_k * 3, len(self.chunks))

        qvec = self.encoder.encode([query], normalize_embeddings=True)
        _, dense_ids = self.index.search(np.asarray(qvec, dtype=np.float32), k_fetch)
        sparse_ids = np.argsort(self.bm25.get_scores(_tokenize(query)))[::-1][:k_fetch]

        scores: dict[int, float] = {}
        for rank, i in enumerate(dense_ids[0]):
            if i != -1:
                scores[int(i)] = scores.get(int(i), 0) + alpha / (60 + rank + 1)
        for rank, i in enumerate(sparse_ids):
            scores[int(i)] = scores.get(int(i), 0) + (1 - alpha) / (60 + rank + 1)

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in best]

    # --- persistence: one dir per agent, index.faiss + chunks.json ---

    def save(self, agent_id: str) -> Path:
        path = INDEX_ROOT / agent_id
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, agent_id: str) -> "HybridRetriever":
        path = INDEX_ROOT / agent_id
        r = cls()
        if not (path / "index.faiss").exists():
            return r
        r.index = faiss.read_index(str(path / "index.faiss"))
        r.chunks = json.loads((path / "chunks.json").read_text())
        r.bm25 = BM25Okapi([_tokenize(c["content"]) for c in r.chunks]) if r.chunks else None
        return r
