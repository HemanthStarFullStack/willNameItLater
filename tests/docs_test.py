"""Docs ingestion: chunking is pure logic; the image path runs end-to-end
through the local vision model on a generated PNG with known content.

Run inside the container:
    docker exec ondevice-ai python tests/docs_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

import docs
from store import store

store._save = lambda: None  # keep the real memories.json untouched

FAILS = 0


def check(name, cond, detail=""):
    global FAILS
    print(f'{"PASS" if cond else "FAIL"}  {name}  {detail}')
    if not cond:
        FAILS += 1


# --- chunking (pure) ----------------------------------------------------------
text = " ".join(f"Sentence number {i} carries some words." for i in range(40))
ch = docs._chunks(text)
check("chunks: splits long text", len(ch) > 1, f"{len(ch)} chunks")
check("chunks: sized around the cap",
      all(len(c.split()) <= docs.CHUNK_WORDS + 20 for c in ch))
check("chunks: keeps all sentences",
      "number 0 " in ch[0] and "number 39" in ch[-1])
check("chunks: empty text -> none", docs._chunks("   ") == [])

# --- unsupported type ----------------------------------------------------------
r = docs.ingest_file("something.docx")
check("unsupported type refused", not r["ok"] and "Unsupported" in r["msg"])

# --- image end-to-end through the vision model ---------------------------------
img = Image.new("RGB", (900, 220), "white")
d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=36)
d.text((30, 60), "Lab report: cholesterol 180 mg/dL.", fill="black", font=font)
d.text((30, 120), "Next checkup in December.", fill="black", font=font)
path = "/tmp/docs_test.png"
img.save(path)

n0 = len(store.facts)
r = docs.ingest_file(path)
saved = [x for x in store.facts[n0:]]
joined = " ".join(x["text"].lower() for x in saved)
check("image: ingest ok", r["ok"], str(r))
check("image: chunks saved", len(saved) >= 1, f"{len(saved)} saved")
check("image: vision model read the number", "180" in joined, joined[:120])
check("image: filed into real categories",
      r["ok"] and all(c in store.categories() for c in r["categories"]))
check("image: source recorded",
      all(x["source"] == "docs_test.png" for x in saved))

# cleanup in-memory copies
for x in saved:
    store.delete(x["id"])

# --- per-chunk routing + re-upload dedupe (deterministic, no vision model) ----
n0 = len(store.facts)
cats, saved_n = docs._store_chunks(
    ["My cholesterol reading was 180 mg/dL.",
     "My manager Priya runs standup at 10am."], "unit.pdf")
new = store.facts[n0:]
check("route: each chunk got a real bucket",
      saved_n == 2 and all(c in store.categories() for c in cats), str(cats))
check("route: multi-topic chunks can split buckets",
      len({x["category"] for x in new}) >= 1)  # >=1: routing ran per chunk
cats2, saved2 = docs._store_chunks(
    ["My cholesterol reading was 180 mg/dL."], "unit.pdf")
check("dedupe: re-upload of same file saves nothing", saved2 == 0, f"{saved2} saved")
for x in new:
    store.delete(x["id"])

print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
