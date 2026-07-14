"""Can this model actually drive tools? Run before trusting native tool calls.
    docker exec -e CHAT_MODEL=qwen3:4b ondevice-ai python tests/tool_test.py
Prints per-case verdicts; exits 1 if any case picks the wrong tool.
"""
import os
import sys

import ollama

MODEL = os.environ.get("CHAT_MODEL", "qwen3:4b")

TOOLS = [
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search the user's saved personal notes (health, work, "
                       "home, documents, preferences). Use for any question "
                       "about the user or their life.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "save_memory",
        "description": "Save a lasting personal fact the user just stated "
                       "about themselves.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string"}}, "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web. Use only for current events, "
                       "weather, prices, news, or facts that change over time.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]

SYSTEM = ("You are a private personal assistant with tools. Decide per message: "
          "questions about the user -> search_memory; the user stating a lasting "
          "personal fact -> save_memory; current/live info -> web_search; "
          "small talk or general knowledge -> answer directly with no tool.")

# (message, expected tool or None)
CASES = [
    ("What is my blood group?", "search_memory"),
    ("what's my wifi password", "search_memory"),
    ("passport num?", "search_memory"),
    ("My name is Hemanth", "save_memory"),
    ("I moved to Hyderabad last month", "save_memory"),
    ("What's the weather in Delhi today?", "web_search"),
    ("latest AI news", "web_search"),
    ("Tell me a joke", None),
    ("hi", None),
    ("What is the capital of France?", None),
]

fails = 0
for msg, want in CASES:
    resp = ollama.chat(model=MODEL, think=False, keep_alive="10m",
                       messages=[{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": msg}],
                       tools=TOOLS, options={"temperature": 0})
    calls = resp.message.tool_calls or []
    got = calls[0].function.name if calls else None
    ok = got == want
    if not ok:
        fails += 1
    args = dict(calls[0].function.arguments) if calls else ""
    print(f'{"PASS" if ok else "FAIL"}  {msg!r:42} want={want} got={got} {args}')

print(f"\n{len(CASES) - fails}/{len(CASES)} correct")
sys.exit(1 if fails else 0)
