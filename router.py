"""The 'parent' router: decides which expert memory a note or question belongs to."""
import llm
from config import CATEGORIES


def route_ingest(text):
    """Pick the best category for a new note (may propose a new one)."""
    cats = ", ".join(CATEGORIES)
    out = llm.chat_json(
        f"Classify this note into exactly ONE category from: [{cats}]. "
        f"If none truly fit, propose a short new category name.\n\n"
        f'Note: "{text}"\n\n'
        'Respond JSON: {"category": "<name>"}',
        system="You are a precise classifier. Output only JSON.",
    )
    cat = (out.get("category") or "Personal").strip()
    return cat[:30] or "Personal"


def route_query(query):
    """Pick which expert memory should answer; None means search everything.

    ponytail: centroid cosine instead of an LLM call — the parent compares the
    query embedding against each expert's average embedding. Deterministic and
    ~free; an LLM router costs seconds per turn and a 1.7B model misroutes.
    Only routes when one expert clearly wins; unsure -> ALL (always safe,
    because the caller falls back to a global search anyway).
    """
    import numpy as np
    from store import store

    cats = {}
    for x in store.facts:
        cats.setdefault(x["category"], []).append(x["embedding"])
    if len(cats) < 2:
        return None

    q = np.array(llm.embed(query), dtype=float)
    q /= (np.linalg.norm(q) or 1.0)
    scored = []
    for cat, embs in cats.items():
        c = np.mean(np.array(embs, dtype=float), axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        scored.append((float(np.dot(q, c)), cat))
    scored.sort(reverse=True)

    best, runner_up = scored[0], scored[1]
    if best[0] >= 0.45 and best[0] - runner_up[0] >= 0.05:
        return best[1]
    return None
