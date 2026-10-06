# Agent eval suite

Spec section 8.2. Each case in `cases.py` is a query plus expected behaviour; `test_agent_quality.py`
drives the LangGraph agent directly (`build_graph` + `chat`, in-memory `HybridRetriever`, no HTTP).

## Run

```bash
uv run pytest tests/eval -q                                   # scripted: FakeLLM replays each case's `scripted` messages
AGENTFORGE_EVAL_LIVE=1 ANTHROPIC_API_KEY=... uv run pytest tests/eval -q   # live: real model (or OPENAI_API_KEY)
```

Scripted mode checks the harness (retrieval ranks the expected source, the tool node runs the expected
tool with the expected args, events stream the answer). Live mode runs the same assertions against the model.

## Categories

| category   | checks                                                                        |
|------------|-------------------------------------------------------------------------------|
| rag        | `expected_source` is retrieved; answer has `expected_contains`, no refusal    |
| tool       | `expected_tool` called with `expected_tool_args`                              |
| refusal    | answer contains a refusal phrase and none of `must_not_contain`               |
| multistep  | tool call, then an answer grounded in its result (`expected_contains`)        |

## Add a case

Append a dict to `EVAL_CASES` with `category`, `query`, any of `expected_source`, `expected_contains`,
`must_not_contain`, `should_use_tool`, `expected_tool`, `expected_tool_args`, `should_refuse`, and a
`scripted` list of `ai(...)` messages the fake LLM returns in order (one per `generate` round; a tool call
message is followed by the final answer). New documents go in `CORPUS` in `conftest.py`; fake tools next
to `get_order_status`.
