"""EVAL_CASES: query + expected behaviour (spec 8.2 shape) + `scripted` replies for the fake LLM.

Keys: query, expected_source, expected_contains, must_not_contain, should_use_tool, expected_tool,
expected_tool_args, should_refuse, category, scripted.
"""

from langchain_core.messages import AIMessage

REFUSAL = "I don't have information about that."


def ai(text="", tool_calls=()):
    return AIMessage(content=text, tool_calls=list(tool_calls),
                     usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def call(name, **args):
    return {"name": name, "args": args, "id": f"call_{name}"}


EVAL_CASES = [
    # --- RAG accuracy ----------------------------------------------------------
    {"category": "rag", "query": "What is the return policy?",
     "expected_source": "returns_policy.txt", "expected_contains": ["30 days", "receipt"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("Items can be returned within 30 days with a receipt [Source 1].")]},
    {"category": "rag", "query": "How long do refunds take to arrive?",
     "expected_source": "returns_policy.txt", "expected_contains": ["5 business days"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("Refunds go to the original payment method within 5 business days [Source 1].")]},
    {"category": "rag", "query": "How do I reset my password?",
     "expected_source": "password_reset.md", "expected_contains": ["Forgot password"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("Click 'Forgot password' on the login page; a reset link is emailed to you [Source 1].")]},
    {"category": "rag", "query": "How long is the password reset link valid?",
     "expected_source": "password_reset.md", "expected_contains": ["24 hours"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("The reset link expires after 24 hours [Source 1].")]},
    {"category": "rag", "query": "How much does express shipping cost and how fast is it?",
     "expected_source": "shipping.txt", "expected_contains": ["$15", "1-2 business days"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("Express shipping costs $15 and takes 1-2 business days [Source 1].")]},
    {"category": "rag", "query": "What does the Team plan cost per month?",
     "expected_source": "pricing.md", "expected_contains": ["$25"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("The Team plan is $25 per user per month [Source 1].")]},
    {"category": "rag", "query": "Do you ship internationally?",
     "expected_source": "shipping.txt", "expected_contains": ["US and Canada"],
     "must_not_contain": [REFUSAL], "should_use_tool": False,
     "scripted": [ai("No. We ship to the US and Canada only [Source 1].")]},

    # --- tool selection --------------------------------------------------------
    {"category": "tool", "query": "Check order ORD-99999 status",
     "should_use_tool": True, "expected_tool": "get_order_status",
     "expected_tool_args": {"order_id": "ORD-99999"},
     "scripted": [ai(tool_calls=[call("get_order_status", order_id="ORD-99999")]),
                  ai("Order ORD-99999 has shipped via UPS, ETA 2026-10-09.")]},
    {"category": "tool", "query": "Where is my order ORD-12345?",
     "should_use_tool": True, "expected_tool": "get_order_status",
     "expected_tool_args": {"order_id": "ORD-12345"},
     "scripted": [ai(tool_calls=[call("get_order_status", order_id="ORD-12345")]),
                  ai("Order ORD-12345 is still processing, ETA 2026-10-12.")]},
    {"category": "tool", "query": "What is 17 * 23?",
     "should_use_tool": True, "expected_tool": "calculate",
     "expected_tool_args": {"expression": "17 * 23"}, "expected_contains": ["391"],
     "scripted": [ai(tool_calls=[call("calculate", expression="17 * 23")]), ai("17 * 23 = 391.")]},
    {"category": "tool", "query": "What is the current date and time in UTC?",
     "should_use_tool": True, "expected_tool": "get_current_time",
     "scripted": [ai(tool_calls=[call("get_current_time")]), ai("The current UTC time is 2026-10-06T12:00:00+00:00.")]},
    {"category": "tool", "query": "Compute (100 - 20) / 4",
     "should_use_tool": True, "expected_tool": "calculate",
     "expected_tool_args": {"expression": "(100 - 20) / 4"}, "expected_contains": ["20"],
     "scripted": [ai(tool_calls=[call("calculate", expression="(100 - 20) / 4")]), ai("(100 - 20) / 4 = 20.0")]},

    # --- hallucination / out-of-scope refusal ------------------------------------
    {"category": "refusal", "query": "What's the weather today?",
     "should_refuse": True, "should_use_tool": False,
     "must_not_contain": ["Based on the documentation", "sunny", "rain"],
     "scripted": [ai(REFUSAL)]},
    {"category": "refusal", "query": "Who is the CEO of Acme Store?",
     "should_refuse": True, "should_use_tool": False, "must_not_contain": ["CEO is"],
     "scripted": [ai(REFUSAL)]},
    {"category": "refusal", "query": "What is the warranty period on headphones?",
     "should_refuse": True, "should_use_tool": False, "must_not_contain": ["1 year", "2 years", "warranty is"],
     "scripted": [ai(REFUSAL)]},
    {"category": "refusal", "query": "Do you ship to Australia and how much does it cost?",
     "expected_source": "shipping.txt", "should_use_tool": False,
     "expected_contains": ["US and Canada"], "must_not_contain": ["$30", "Australia costs"],
     "scripted": [ai("We ship to the US and Canada only, so shipping to Australia is not available [Source 1].")]},

    # --- multi-step: tool then grounded answer -----------------------------------
    {"category": "multistep", "query": "Has order ORD-99999 shipped yet, and which carrier?",
     "should_use_tool": True, "expected_tool": "get_order_status",
     "expected_tool_args": {"order_id": "ORD-99999"}, "expected_contains": ["shipped", "UPS"],
     "scripted": [ai(tool_calls=[call("get_order_status", order_id="ORD-99999")]),
                  ai("Yes, ORD-99999 has shipped via UPS, ETA 2026-10-09.")]},
    {"category": "multistep", "query": "What is 15% of 200?",
     "should_use_tool": True, "expected_tool": "calculate", "expected_contains": ["30"],
     "scripted": [ai(tool_calls=[call("calculate", expression="200 * 0.15")]), ai("15% of 200 is 30.")]},
    {"category": "multistep", "query": "What would 3 Team plan seats cost per month?",
     "expected_source": "pricing.md", "should_use_tool": True, "expected_tool": "calculate",
     "expected_contains": ["75"],
     "scripted": [ai(tool_calls=[call("calculate", expression="25 * 3")]),
                  ai("The Team plan is $25 per user per month [Source 1], so 3 seats cost $75 per month.")]},
]
