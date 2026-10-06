"""Usage:
  python -m agentforge.cli ingest <agent_id> <url-or-path> [...]
  python -m agentforge.cli query  <agent_id> "<question>" [--k 5] [--alpha 0.7]
"""

import argparse

from agentforge.ingestion import ingest
from agentforge.retriever import HybridRetriever


def main():
    p = argparse.ArgumentParser(prog="agentforge")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest")
    i.add_argument("agent_id")
    i.add_argument("sources", nargs="+")

    q = sub.add_parser("query")
    q.add_argument("agent_id")
    q.add_argument("question")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--alpha", type=float, default=0.7)

    a = p.parse_args()
    r = HybridRetriever.load(a.agent_id)

    if a.cmd == "ingest":
        seen = {c["source"] for c in r.chunks}
        for src in a.sources:
            if src in seen:
                print(f"skip {src} (already indexed)")
                continue
            h, chunks = ingest(src)
            r.add(chunks)
            print(f"{src}: {len(chunks)} chunks  sha256={h[:12]}")
        print(f"saved -> {r.save(a.agent_id)}  ({len(r.chunks)} chunks total)")
    else:
        for chunk, score in r.search(a.question, top_k=a.k, alpha=a.alpha):
            page = f" p.{chunk['page']}" if chunk["page"] else ""
            print(f"[{score:.4f}] {chunk['source']}{page}\n  {chunk['content'][:200]!r}\n")


if __name__ == "__main__":
    main()
