"""Spot-check answer style: natural second-person sentences, no literal notes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline
from store import store

store._save = lambda: None

QUERIES = [
    "What is my blood group?",
    "what's my wifi password",
    "passport num",
    "What is my name?",
    "whats my passport number and who is my manager",
    "am I allergic to anything",
]

notes = {x["text"].strip().lower() for x in store.facts}
bad = 0
for q in QUERIES:
    reply = pipeline.chat(q, []).split("\n")[0].strip()
    literal = reply.lower().rstrip(".") in {n.rstrip(".") for n in notes}
    first_person = reply.lower().startswith(("my ", "i'm ", "i "))
    flag = " <-- LITERAL NOTE" if literal else (" <-- FIRST PERSON" if first_person else "")
    if literal or first_person:
        bad += 1
    print(f"{q!r:55} -> {reply}{flag}")

print(f"\n{'STYLE OK' if bad == 0 else f'{bad} replies still off'}")
sys.exit(1 if bad else 0)
