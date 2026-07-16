"""Can this model actually drive tools? Run before trusting native tool calls.
    docker exec -e CHAT_MODEL=qwen3:4b ondevice-ai python tests/tool_test.py
Prints per-case verdicts; exits 1 if any case picks the wrong tool.
"""
import os
import sys
import time

import ollama

MODEL = os.environ.get("CHAT_MODEL", "qwen3:4b")
# Must match how the app loads the model. Qwen3.5 defaults to a 256K window,
# which alone blows the KV cache past this GPU (measured: 23GB "size", 73% on
# CPU). The app pins 4096, so the bench has to as well or it measures nothing.
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))

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
times = []
for msg, want in CASES:
    t0 = time.time()
    resp = ollama.chat(model=MODEL, think=False, keep_alive="10m",
                       messages=[{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": msg}],
                       tools=TOOLS,
                       options={"temperature": 0, "num_ctx": NUM_CTX})
    dt = time.time() - t0
    times.append(dt)
    calls = resp.message.tool_calls or []
    got = calls[0].function.name if calls else None
    ok = got == want
    if not ok:
        fails += 1
    args = dict(calls[0].function.arguments) if calls else ""
    print(f'{"PASS" if ok else "FAIL"}  {dt:6.1f}s  {msg!r:42} '
          f'want={want} got={got} {args}')

# Correctness is only half the question: a model that picks the right tool but
# spills to CPU is unusable. qwen3:4b scored 10/10 here at ~198s/answer.
print(f"\n{MODEL}: {len(CASES) - fails}/{len(CASES)} correct · "
      f"median {sorted(times)[len(times)//2]:.1f}s · slowest {max(times):.1f}s")
sys.exit(1 if fails else 0)
