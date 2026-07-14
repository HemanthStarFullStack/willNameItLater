"""On-device memory store: hybrid (keyword + vector) search over saved notes.

Each memory is a dict: {id, category, text, source, embedding}.
Persisted to a plain JSON file so it survives restarts.
"""
import os
import json
import threading

import numpy as np
from rank_bm25 import BM25Okapi

import llm
from config import DATA_DIR, DATA_FILE


def _tok(s):
    return s.lower().split()


class Store:
    def __init__(self):
        self.facts = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                self.facts = json.load(f)

    def _save(self):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)

    def add(self, text, category, source=""):
        with self._lock:
            emb = llm.embed(text)
            fid = (max(x["id"] for x in self.facts) + 1) if self.facts else 1
            self.facts.append({"id": fid, "category": category,
                               "text": text, "source": source,
                               "embedding": emb})
            self._save()
            return fid

    def update(self, fid, text):
        with self._lock:
            for x in self.facts:
                if x["id"] == fid:
                    x["text"] = text
                    x["embedding"] = llm.embed(text)
                    self._save()
                    return True
            return False

    def delete(self, fid):
        with self._lock:
            n = len(self.facts)
            self.facts = [x for x in self.facts if x["id"] != fid]
            if len(self.facts) < n:
                self._save()
                return True
            return False

    def categories(self):
        c = {}
        for x in self.facts:
            c[x["category"]] = c.get(x["category"], 0) + 1
        return c

    def facts_in(self, category=None):
        return [x for x in self.facts
                if category is None or x["category"] == category]

    def search(self, query, category=None, k=8, with_score=False):
        """Hybrid retrieval: fuse vector-cosine and BM25 rankings via RRF.

        With with_score=True, also returns the best raw cosine similarity — a
        calibrated signal for "is any saved note actually relevant?" that the
        chat layer uses to decide whether to invoke RAG at all.
        """
        pool = self.facts_in(category) or self.facts
        if not pool:
            return ([], 0.0) if with_score else []

        q = np.array(llm.embed(query), dtype=float)
        qn = np.linalg.norm(q) or 1.0
        cosines = {}
        for x in pool:
            a = np.array(x["embedding"], dtype=float)
            cosines[x["id"]] = float(np.dot(q, a) / (qn * (np.linalg.norm(a) or 1.0)))

        vec_ranked = sorted(pool, key=lambda x: cosines[x["id"]], reverse=True)

        bm = BM25Okapi([_tok(x["text"]) for x in pool])
        scores = bm.get_scores(_tok(query))
        bm_ranked = [pool[i] for i in np.argsort(scores)[::-1]]

        # Reciprocal Rank Fusion
        rr = {}
        for rank, x in enumerate(vec_ranked):
            rr[x["id"]] = rr.get(x["id"], 0.0) + 1.0 / (60 + rank)
        for rank, x in enumerate(bm_ranked):
            rr[x["id"]] = rr.get(x["id"], 0.0) + 1.0 / (60 + rank)

        fused = sorted(pool, key=lambda x: rr.get(x["id"], 0.0), reverse=True)[:k]
        # Copies with the raw cosine attached, so callers can drop distractor
        # notes without these extra keys leaking into the persisted facts.
        out = [dict(x, _cos=cosines[x["id"]]) for x in fused]
        if with_score:
            top = max((h["_cos"] for h in out), default=0.0)
            return out, top
        return out


store = Store()
