"""LangGraph agent: router -> retrieve -> generate <-> tools -> finalize.

Conversation memory = LangGraph checkpointer keyed by thread_id (conversation id).
Retrieved context is injected transiently in `generate`, never persisted into messages.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentforge.llm import calculate_cost, invoke_with_failover

MAX_TOOL_ROUNDS = 8
# ponytail: in-memory checkpointer, lost on restart. Swap for PostgresSaver when persistence matters.
CHECKPOINTER = MemorySaver()


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    retrieved: list[dict]
    rounds: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_used: str


def _context_message(chunks: list[dict]) -> SystemMessage:
    parts = [f"[Source {i + 1}: {c['source']}{f' p.{c['page']}' if c.get('page') else ''}]\n{c['content']}"
             for i, c in enumerate(chunks)]
    return SystemMessage(content="Retrieved context from the knowledge base:\n\n" + "\n\n---\n\n".join(parts)
                         + "\n\nUse it to answer; cite sources by number. If it is irrelevant, say you don't know.")


def build_graph(cfg: dict, retriever, tools: list):
    """cfg: {system_prompt, llm_provider, llm_model, temperature, top_k, hybrid_alpha}."""

    def route(state: State) -> str:
        # Retrieve on every fresh user turn when an index exists; tool follow-ups skip it.
        last = state["messages"][-1]
        return "retrieve" if isinstance(last, HumanMessage) and retriever and retriever.chunks else "generate"

    def retrieve(state: State) -> State:
        hits = retriever.search(state["messages"][-1].content, top_k=cfg.get("top_k", 5),
                                alpha=cfg.get("hybrid_alpha", 0.7))
        return {"retrieved": [{**c, "score": round(s, 4)} for c, s in hits], "rounds": 0}

    async def generate(state: State) -> State:
        msgs = [SystemMessage(content=cfg.get("system_prompt", "You are a helpful assistant."))]
        if state.get("retrieved"):
            msgs.append(_context_message(state["retrieved"]))
        msgs += state["messages"]
        rounds = state.get("rounds", 0)
        try:
            resp, _, model = await invoke_with_failover(
                cfg.get("llm_provider", "anthropic"), cfg.get("llm_model", "claude-sonnet-4-5"), msgs,
                tools=tools if rounds < MAX_TOOL_ROUNDS else None, temperature=cfg.get("temperature", 0.3))
        except Exception as e:  # noqa: BLE001 - every provider failed; degrade, don't crash the turn
            return {"messages": [AIMessage(content=f"Sorry, I couldn't reach the language model ({type(e).__name__}). Please try again.")]}
        usage = resp.usage_metadata or {}
        return {"messages": [resp], "rounds": rounds + 1, "model_used": model,
                "input_tokens": state.get("input_tokens", 0) + usage.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0)}

    def after_generate(state: State) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "finalize"

    def finalize(state: State) -> State:
        return {"cost_usd": calculate_cost(state.get("model_used", ""), state.get("input_tokens", 0),
                                           state.get("output_tokens", 0))}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("finalize", finalize)
    g.add_conditional_edges(START, route, {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "generate")
    if tools:
        # handle_tool_errors=True: any exception becomes a ToolMessage the model reads and recovers from
        g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        g.add_conditional_edges("generate", after_generate, {"tools": "tools", "finalize": "finalize"})
        g.add_edge("tools", "generate")
    else:
        g.add_edge("generate", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=CHECKPOINTER)


async def chat(graph, conversation_id: str, message: str):
    """Yield SSE-shaped (event, data) tuples for one user turn."""
    config = {"configurable": {"thread_id": conversation_id}}
    streamed = False
    async for ev in graph.astream_events({"messages": [HumanMessage(content=message)]}, config, version="v2"):
        kind, name, data = ev["event"], ev.get("name"), ev["data"]
        if kind == "on_chain_end" and name == "retrieve":
            yield "retrieval", {"chunks": [{k: c[k] for k in ("source", "page", "score")} | {"content": c["content"][:200]}
                                           for c in data["output"]["retrieved"]]}
        elif kind == "on_chat_model_stream":
            c = data["chunk"].content
            text = c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))
            if text:
                streamed = True
                yield "token", {"content": text}
        elif kind == "on_tool_start":
            yield "tool_call", {"name": name, "args": data.get("input"), "status": "executing"}
        elif kind == "on_tool_end":
            out = data.get("output")
            yield "tool_result", {"name": name, "result": str(getattr(out, "content", out))[:500]}
        elif kind == "on_tool_error":
            yield "tool_result", {"name": name, "result": f"error: {data.get('error')}"[:500], "error": True}
    s = (await graph.aget_state(config)).values
    if not streamed and s.get("messages"):  # non-streaming provider or the LLM-failure fallback message
        yield "token", {"content": s["messages"][-1].text}
    yield "done", {k: s.get(k, 0) for k in ("input_tokens", "output_tokens", "cost_usd")} | {"model": s.get("model_used")}


def history(graph, conversation_id: str) -> list[dict]:
    """Messages for a conversation from the checkpointer, in API shape."""
    state = graph.get_state({"configurable": {"thread_id": conversation_id}})
    out = []
    for m in state.values.get("messages", []):
        role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(m.type, m.type)
        d = {"id": m.id, "role": role, "content": m.content if isinstance(m.content, str) else m.text}
        if role == "assistant" and getattr(m, "tool_calls", None):
            d["tool_calls"] = m.tool_calls
        out.append(d)
    return out
