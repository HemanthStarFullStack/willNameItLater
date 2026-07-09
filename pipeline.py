"""The end-to-end brain: ingest() saves & routes; ask() retrieves, grades,
answers, and verifies. Every step records a trace so the UI can show its work.
"""
import llm
import router
from store import store
from config import SEARCH_K, TOP_K


def ingest(text, source=""):
    text = (text or "").strip()
    if not text:
        return {"ok": False, "msg": "Nothing to save."}
    category = router.route_ingest(text)
    fid = store.add(text, category, source)
    return {"ok": True, "category": category, "id": fid}


def _fmt(hits):
    return "\n".join(f'[{i+1}] ({h["category"]}) {h["text"]}'
                     for i, h in enumerate(hits))


def _crag(query, hits):
    """Corrective-RAG grade: are the retrieved notes good enough to answer?"""
    if not hits:
        return {"verdict": "weak", "reason": "no results"}
    out = llm.chat_json(
        f'Question: "{query}"\n\nRetrieved notes:\n{_fmt(hits)}\n\n'
        "Are these notes relevant AND sufficient to answer the question? "
        'Respond JSON: {"verdict": "good|weak", "reason": "..."}',
        system="You grade retrieval quality. Output only JSON.",
    )
    return out or {"verdict": "good", "reason": ""}


def _verify(answer, context):
    """Groundedness gate: is every claim backed by the notes?"""
    if "don't have that in your data" in answer.lower():
        return {"grounded": True, "reason": "honest no-answer"}
    out = llm.chat_json(
        f"Notes:\n{context}\n\nAnswer:\n{answer}\n\n"
        "The answer is GROUNDED if all of its facts come from the notes "
        "(rephrasing or summarizing is fine). It is NOT grounded ONLY if it "
        "adds a fact that is absent from the notes, or contradicts them. "
        "Do not nitpick wording. "
        'Respond JSON: {"grounded": true, "reason": "..."}',
        system="You detect invented facts. Be lenient about phrasing. Output only JSON.",
    )
    return out if "grounded" in out else {"grounded": True, "reason": ""}


def ask(query):
    query = (query or "").strip()
    trace = {}
    if not query:
        return {"answer": "Ask me something about your data.", "trace": trace}

    category = router.route_query(query)
    trace["routed_to"] = category or "ALL"

    hits = store.search(query, category=category, k=SEARCH_K)
    grade = _crag(query, hits)
    trace["crag"] = grade

    # Corrective step: if retrieval was weak, rewrite and search everything once.
    if grade.get("verdict") == "weak":
        rewrite = llm.chat(
            f'Rewrite this as a short keyword search query:\n"{query}"')
        trace["rewritten_query"] = rewrite
        retry = store.search(rewrite, category=None, k=SEARCH_K)
        if retry:
            hits = retry
            trace["crag_retry"] = _crag(query, hits)

    hits = hits[:TOP_K]
    trace["retrieved"] = [{"category": h["category"], "text": h["text"]}
                          for h in hits]
    context = _fmt(hits) if hits else "(no notes found)"

    answer = llm.chat(
        "Answer the question using ONLY the notes below. If the answer is not "
        'in them, reply exactly: "I don\'t have that in your data yet."\n\n'
        f"Notes:\n{context}\n\nQuestion: {query}",
        system="You answer strictly from the user's notes. Never invent facts.",
    )

    verdict = _verify(answer, context)
    trace["verify"] = verdict
    if not verdict.get("grounded", True):
        answer = ("🤔 I can't back that up from your data, so I won't guess.\n\n"
                  f"_(unverified draft: {answer})_")

    return {"answer": answer, "trace": trace}
