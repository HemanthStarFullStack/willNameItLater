"""The end-to-end brain: ingest() saves & routes; ask() retrieves, grades,
answers, and verifies. Every step records a trace so the UI can show its work.
"""
import re

import llm
import router
import web
from store import store
from config import SEARCH_K, TOP_K, RAG_THRESHOLD

# Obvious "needs the live web" signals. The 1.5B model can't reliably sense its
# own knowledge cutoff (it will confidently guess "who won the 2026 ..."), so we
# force a search on these cues and fall back to an LLM judge for everything else.
_WEB_HINTS = re.compile(
    r"\b(latest|current(ly)?|today|tonight|now|recent(ly)?|this (week|month|year)|"
    r"news|headline|price|prices|cost of|weather|forecast|score|scores|result|"
    r"results|who won|winner|standings|stock|shares|released?|launch(ed|ing)?|"
    r"update|version|20(2[4-9]|3\d))\b", re.I)

# Words too common to signal grounding either way.
_STOP = set(
    "a an the is are was were am i my me you your of to in on at for and or with "
    "that this it its as be been being have has had do does did what when where "
    "who why how which not no yes can will would should may might".split()
)


def _content_words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


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
    """Groundedness gate. Answers here are extractive, so groundedness is mostly
    "do the answer's words come from the notes?" — a fast, deterministic check.
    We only spend an LLM call when the answer introduces terms not in the notes,
    which is exactly where a hallucination would show up.
    """
    if "don't have that in your data" in answer.lower():
        return {"grounded": True, "reason": "honest no-answer"}

    words = _content_words(answer)
    ctx = set(_content_words(context))
    covered = sum(w in ctx for w in words) / len(words) if words else 1.0

    if covered >= 0.6:
        return {"grounded": True, "reason": f"lexical coverage {covered:.0%}"}

    # Low overlap -> the answer added new terms; let the model judge.
    out = llm.chat_json(
        f"NOTES:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Does every fact in ANSWER also appear in NOTES? Rephrasing is fine. "
        'Answer false ONLY if ANSWER states a fact that is nowhere in NOTES. '
        'JSON: {"grounded": true}',
        system="Fact-check the answer against the notes. When unsure, say true. Output only JSON.",
    )
    out = out if "grounded" in out else {"grounded": True}
    out.setdefault("reason", f"lexical coverage {covered:.0%}")
    return out


# Pure greetings/small talk carry no data intent — answer them instantly instead
# of running the full retrieval pipeline (and tripping the groundedness gate).
_GREETINGS = {"hi", "hii", "hey", "yo", "hello", "hello there", "hey there",
              "sup", "hola", "howdy", "good morning", "good afternoon",
              "good evening", "hi there", "thanks", "thank you", "ok", "okay"}


def ask(query):
    query = (query or "").strip()
    trace = {}
    if not query:
        return {"answer": "Ask me something about your data.", "trace": trace}

    if re.sub(r"[^a-z ]", "", query.lower()).strip() in _GREETINGS:
        trace["routed_to"] = "greeting"
        return {"answer": "Hi! Ask me anything about your saved data.",
                "trace": trace}

    # Search across all memories. Routing to a single category first would add a
    # slow extra LLM call and can miss on small data — fusing over everything is
    # faster and more robust for the prototype.
    trace["routed_to"] = "ALL"
    hits = store.search(query, category=None, k=SEARCH_K)
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
        f"NOTES about the user:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer the question directly and briefly using the NOTES. "
        "If a note is relevant, use it even if the wording differs. "
        "Do not greet, chat, or add anything not asked. "
        'If NOTES contain nothing relevant (including for greetings or small talk), '
        'reply exactly: "I don\'t have that in your data yet."',
        system="You answer strictly from the user's notes. Never invent facts. No chit-chat.",
    )

    verdict = _verify(answer, context)
    trace["verify"] = verdict
    if not verdict.get("grounded", True):
        answer = ("🤔 I can't back that up from your data, so I won't guess.\n\n"
                  f"_(unverified draft: {answer})_")

    return {"answer": answer, "trace": trace}


# --- Chat entry point -------------------------------------------------------
# A conversation doesn't need RAG on every turn. We decide per message: if a
# saved note is actually relevant (retrieval score over threshold) we answer
# from memory; otherwise we just reply like a normal assistant. One decision,
# one LLM call per turn.

def _recent(history, n=4):
    """Flatten the last few turns of gr chat history into plain text."""
    out = []
    for h in (history or [])[-n:]:
        if isinstance(h, dict):
            out.append(f'{h.get("role", "user").capitalize()}: {h.get("content", "")}')
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            out.append(f"User: {h[0]}\nAssistant: {h[1]}")
    return "\n".join(out)


# Messages that can carry a durable personal fact look like "my X is Y" /
# "I am/like/work..." and are not questions or commands. Only those pay the
# extra extraction LLM call; everything else skips it.
_FACT_HINT = re.compile(
    r"\b(my|mine|i am|i'm|im|i have|i've|i work|i live|i like|i love|i hate|"
    r"i prefer|i was|i use|i drive|i own|call me)\b", re.I)
_NOT_FACT_START = re.compile(
    r"^(what|who|whom|whose|when|where|why|how|is|are|am|was|were|do|does|did|"
    r"can|could|will|would|should|shall|may|might|tell|show|give|list|search|"
    r"find|explain|write|make|create|help|please)\b", re.I)


def _looks_like_fact(message):
    if "?" in message:
        return False
    if _NOT_FACT_START.match(message.strip()):
        return False
    return bool(_FACT_HINT.search(message))


def _maybe_remember(message):
    """If the user just told us a lasting personal fact, keep it.
    Returns a footer label ("Work" / "Work · updated") or None."""
    if not _looks_like_fact(message):
        return None
    out = llm.chat_json(
        f'Message from the user: "{message}"\n\n'
        "If this states a lasting personal fact about the user (name, health, "
        "work, home, preferences, possessions, relationships, plans), rewrite "
        'it as one short first-person note, e.g. "My name is Hemanth." '
        "If it is a question, a request, or small talk, use null.\n"
        'JSON: {"fact": "<note>"}  or  {"fact": null}',
        system="You extract durable personal facts. Output only JSON.",
    )
    fact = out.get("fact")
    fact = fact.strip() if isinstance(fact, str) else ""
    if not fact:
        return None

    # ponytail: cosine >= 0.92 means the same fact restated or corrected ->
    # update in place instead of piling up duplicates. Distinct facts about the
    # same topic (allergy A vs allergy B) score lower and are added separately.
    hits, score = store.search(fact, k=1, with_score=True)
    if hits and score >= 0.92:
        store.update(hits[0]["id"], fact)
        return f'{hits[0]["category"]} · updated'
    r = ingest(fact, "chat")
    return r["category"] if r.get("ok") else None


def chat(message, history=None):
    """Return a reply string. Routes the question to the right expert memory
    when a saved note is relevant, falls back to web/general chat otherwise,
    and quietly saves any new personal fact the user states."""
    message = (message or "").strip()
    if not message:
        return "Ask me something."

    steps = []
    expert = router.route_query(message)
    steps.append(f"routed to <b>{expert or 'all'}</b> memories")
    hits, score = store.search(message, category=expert, k=TOP_K,
                               with_score=True)
    if expert and score < RAG_THRESHOLD:
        # Routed to the wrong expert? Check all memories before giving up.
        expert = None
        hits, score = store.search(message, category=None, k=TOP_K,
                                   with_score=True)
        steps.append(f"expert scored low → re-searched <b>all</b> ({score:.2f})")

    if score >= RAG_THRESHOLD and hits:
        # ponytail: only near-top notes reach the model. A 1.7B answerer given
        # loosely-related notes quotes the wrong one (asked passport, said the
        # name). Measured: right note ~0.67-0.77, distractors ~0.35-0.45.
        # Ceiling: multi-fact questions lose secondary notes; widen if needed.
        hits = [h for h in hits if h.get("_cos", 1.0) >= score - 0.12]
        steps.append(f"match {score:.2f} ≥ {RAG_THRESHOLD} → memory path, "
                     f"kept {len(hits)} note(s) after dropping distractors")
        context = _fmt(hits)
        answer = llm.chat(
            f"NOTES about the user:\n{context}\n\n"
            f"QUESTION: {message}\n\n"
            "Answer directly and briefly using the NOTES. Use a note even if the "
            "wording differs. Never repeat or rephrase the question — state the "
            "actual value or fact from the NOTES. Do not add anything not asked. "
            "Do not include reference numbers like [1] or category tags. If the "
            "NOTES do not actually contain the answer, reply exactly: \"I don't "
            'have that in your data yet." — never substitute a different fact.',
            system="You answer strictly from the user's notes. Never invent facts.",
        )
        verdict = _verify(answer, context)
        steps.append(f"verify: {'✔ grounded' if verdict.get('grounded', True) else '✘ unverified'}"
                     f" ({verdict.get('reason', '')})")
        if not verdict.get("grounded", True):
            answer += "\n\n_(couldn't fully verify this against your notes)_"
        which = expert or hits[0]["category"]
        footer = f"🧠 {which} memory · match {score:.2f}"
    else:
        if _needs_web(message):
            steps.append(f"no memory match ({score:.2f}), live-info cue → web search")
        else:
            steps.append(f"no memory match ({score:.2f}), no web cue → general chat")
        answer, used_web = _chat_or_search(message, history)
        if used_web:
            footer = f"🌐 web search · no memory match ({score:.2f})"
        else:
            footer = f"💬 general reply · no memory match ({score:.2f})"

    saved = _maybe_remember(message)
    if saved:
        footer += f" · 💾 saved to {saved}"
        steps.append(f"💾 new fact detected → saved to {saved}")

    trace = "<br>".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return (f"{answer}\n\n<details><summary>🧭 thought process</summary>"
            f"<sub>{trace}</sub></details><sub>{footer}</sub>")


def _needs_web(message):
    """Web lookup only on live-info keyword cues. An LLM judge was tried but a
    small model can't do this reliably (no-think says 'web' to everything), and
    the cues already cover the real cases. ponytail: regex beats a flaky call."""
    return bool(_WEB_HINTS.search(message))


def _chat_or_search(message, history):
    """Normal chat, escalating to a web search when the question needs live info.
    Returns (answer, used_web)."""
    if not _needs_web(message):
        convo = _recent(history)
        prompt = f"{convo}\nUser: {message}\nAssistant:" if convo else message
        answer = llm.chat(prompt, system="You are a helpful, concise personal "
                          "assistant. Answer in a sentence or two.")
        return answer, False

    results = web.search(message, k=4)
    if not results:
        return "I tried to search the web but couldn't reach it right now.", True
    answer = llm.chat(
        f"Web results:\n{web.format_results(results)}\n\n"
        f"Question: {message}\n\n"
        "Answer the question using the web results above, briefly. State actual "
        "values from the results; never write placeholders like [insert ...].",
        system="You answer from the provided web results. Be concise and factual.",
    )
    answer += "\n\nSources: " + ", ".join(
        f'[{i+1}]({r["url"]})' for i, r in enumerate(results))
    return answer, True
