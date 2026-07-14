"""Edge-case checks for the multi-RAG pipeline. Integration test — talks to
the real Ollama, never writes to disk (store._save is stubbed).

Run inside the container:
    docker exec ondevice-ai python tests/edge_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline
import router
from store import store

store._save = lambda: None  # keep the real memories.json untouched

FAILS = 0


def check(name, cond, detail=""):
    global FAILS
    print(f'{"PASS" if cond else "FAIL"}  {name}  {detail}')
    if not cond:
        FAILS += 1


# Ensure a queryable baseline even on a fresh store.
if not store.facts:
    store.add("My blood group is O positive.", "Health")
    store.add("Home wifi password is bluefalcon77.", "Home")
    store.add("My manager is Priya; standup is 10am daily on weekdays.", "Work")

# --- pure heuristics (no LLM) ------------------------------------------------
check("web-hint: 'latest news' triggers", pipeline._needs_web("latest news on AI"))
check("web-hint: memory Q does not", not pipeline._needs_web("what is my blood group"))
check("fact-hint: statement passes", pipeline._looks_like_fact("My name is Hemanth"))
check("fact-hint: question blocked", not pipeline._looks_like_fact("What is my name?"))
check("fact-hint: command blocked", not pipeline._looks_like_fact("tell me a joke"))
check("fact-hint: greeting blocked", not pipeline._looks_like_fact("hi"))
check("fact-hint: no-question-mark question blocked",
      not pipeline._looks_like_fact("what's my wifi password"))

# --- store update/delete roundtrip -------------------------------------------
n0 = len(store.facts)
fid = store.add("Temporary test fact about kayaking.", "Personal")
check("store.add", len(store.facts) == n0 + 1)
old_emb = next(x["embedding"] for x in store.facts if x["id"] == fid)
store.update(fid, "Temporary test fact about mountain biking.")
new = next(x for x in store.facts if x["id"] == fid)
check("store.update text", "biking" in new["text"])
check("store.update re-embeds", new["embedding"] != old_emb)
check("store.delete", store.delete(fid) and len(store.facts) == n0)
check("store.delete missing id is False", not store.delete(99999))

# --- routing -----------------------------------------------------------------
r = router.route_query("What is my blood group?")
check("router: blood group -> Health or ALL", r in (None, "Health"), f"got {r!r}")

# --- chat paths (LLM-backed) ---------------------------------------------------
check("chat: empty message", pipeline.chat("") == "Ask me something.")

reply = pipeline.chat("What is my blood group?", [])
check("chat: memory hit answers", "o positive" in reply.lower(), reply[-80:])
check("chat: memory footer", "🧠" in reply)
check("chat: footer names an expert", "memory · match" in reply)

n0 = len(store.facts)
reply = pipeline.chat("Tell me a joke", [])
check("chat: general path", "💬" in reply)
check("chat: joke not saved", len(store.facts) == n0)

reply = pipeline.chat("What's my wifi password?", [])
check("chat: question not saved", len(store.facts) == n0 and "💾" not in reply)

# --- the persistent-memory fix -------------------------------------------------
reply = pipeline.chat("My name is Hemanth", [])
check("chat: fact saved", len(store.facts) == n0 + 1, reply[-80:])
check("chat: save footer", "💾" in reply)
check("chat: fact text kept", any("hemanth" in x["text"].lower()
                                  for x in store.facts))

reply = pipeline.chat("What is my name?", [])
check("chat: recalls saved fact", "hemanth" in reply.lower(), reply[-80:])

n1 = len(store.facts)
pipeline.chat("My name is Hemanth", [])
check("chat: restating dedupes (update, not add)", len(store.facts) == n1)

# --- empty store ----------------------------------------------------------------
snapshot, store.facts = store.facts, []
reply = pipeline.chat("hello there", [])
check("chat: empty store no crash", "💬" in reply)
store.facts = snapshot

print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
