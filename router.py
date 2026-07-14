"""The 'parent' router: decides which expert memory a note or question belongs to."""
import numpy as np

import llm
from config import CATEGORIES
from store import store


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


def _centroids():
    """Unit-normalised mean embedding per existing expert memory."""
    cats = {}
    for x in store.facts:
        cats.setdefault(x["category"], []).append(x["embedding"])
    out = {}
    for cat, embs in cats.items():
        c = np.mean(np.array(embs, dtype=float), axis=0)
        out[cat] = c / (np.linalg.norm(c) or 1.0)
    return out


def _embed_unit(text):
    q = np.array(llm.embed(text), dtype=float)
    return q / (np.linalg.norm(q) or 1.0)


def route_query(query):
    """Pick which expert memory should answer; None means search everything.

    ponytail: centroid cosine instead of an LLM call — the parent compares the
    query embedding against each expert's average embedding. Deterministic and
    ~free; an LLM router costs seconds per turn and a 1.7B model misroutes.
    Only routes when one expert clearly wins; unsure -> ALL (always safe,
    because the caller falls back to a global search anyway).
    """
    cents = _centroids()
    if len(cents) < 2:
        return None
    q = _embed_unit(query)
    scored = sorted(((float(np.dot(q, c)), cat) for cat, c in cents.items()),
                    reverse=True)
    best, runner_up = scored[0], scored[1]
    if best[0] >= 0.45 and best[0] - runner_up[0] >= 0.05:
        return best[1]
    return None


def snap_to_existing(text, proposed):
    """A brand-new bucket must earn its existence: if the note sits close to an
    existing expert's centroid, file it there instead of fragmenting the
    memories ("Fitness" appearing next to "Health")."""
    cents = _centroids()
    if not cents:
        return proposed
    q = _embed_unit(text)
    best_cat, best = proposed, 0.0
    for cat, c in cents.items():
        s = float(np.dot(q, c))
        if s > best:
            best, best_cat = s, cat
    return best_cat if best >= 0.45 else proposed
