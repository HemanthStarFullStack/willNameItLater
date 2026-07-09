"""Thin wrapper around local Ollama: chat, JSON judge, and embeddings."""
import json
import re
import ollama

from config import CHAT_MODEL, EMBED_MODEL, NUM_CTX


def _content(resp):
    try:
        return resp.message.content
    except Exception:
        return resp["message"]["content"]


def chat(prompt, system=None, temperature=0.2):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=CHAT_MODEL, messages=messages,
                       options={"temperature": temperature, "num_ctx": NUM_CTX})
    return _content(resp).strip()


def chat_json(prompt, system=None):
    """Ask for a strict-JSON answer and parse it, with a best-effort fallback."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=CHAT_MODEL, messages=messages,
                       format="json", options={"temperature": 0, "num_ctx": NUM_CTX})
    txt = _content(resp)
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}


def embed(text):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    try:
        return list(resp.embedding)
    except Exception:
        return list(resp["embedding"])
