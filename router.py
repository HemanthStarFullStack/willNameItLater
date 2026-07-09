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
    """Pick which memory holds the answer; None means search everything."""
    cats = ", ".join(CATEGORIES + ["ALL"])
    out = llm.chat_json(
        f"Which single category best holds the answer to this question? "
        f"Options: [{cats}]. Use ALL if unsure.\n\n"
        f'Question: "{query}"\n\n'
        'Respond JSON: {"category": "<name>"}',
        system="You route questions to a knowledge category. Output only JSON.",
    )
    cat = (out.get("category") or "ALL").strip()
    return None if cat.upper() == "ALL" else cat
