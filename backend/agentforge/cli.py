"""Usage:
  python -m agentforge.cli ingest <agent_id> <url-or-path> [...]
  python -m agentforge.cli query  <agent_id> "<question>" [--k 5] [--alpha 0.7]
  python -m agentforge.cli chat   <agent_id>            (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
"""

import argparse
import asyncio
import uuid

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

    ch = sub.add_parser("chat")
    ch.add_argument("agent_id")
    ch.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ch.add_argument("--model", default="claude-sonnet-4-5")

    a = p.parse_args()
    r = HybridRetriever.load(a.agent_id)

    if a.cmd == "chat":
        return asyncio.run(chat_loop(a, r))

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


async def chat_loop(a, r):
    from agentforge.agent import build_graph, chat
    from agentforge.tools import build_tools

    graph = build_graph({"llm_provider": a.provider, "llm_model": a.model}, r, build_tools(r, []))
    cid = str(uuid.uuid4())
    print(f"chatting with {a.agent_id} ({len(r.chunks)} chunks). ctrl-d to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            return
        if not q:
            continue
        async for event, data in chat(graph, cid, q):
            if event == "token":
                print(data["content"], end="", flush=True)
            elif event == "tool_call":
                print(f"\n[tool {data['name']} {data['args']}]", flush=True)
            elif event == "done":
                print(f"\n[{data['model']} in={data['input_tokens']} out={data['output_tokens']} ${data['cost_usd']:.5f}]")


if __name__ == "__main__":
    main()
