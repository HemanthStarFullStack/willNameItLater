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

    def categories(self):
        c = {}
        for x in self.facts:
            c[x["category"]] = c.get(x["category"], 0) + 1
        return c

    def facts_in(self, category=None):
        return [x for x in self.facts
                if category is None or x["category"] == category]

    def search(self, query, category=None, k=8):
        """Hybrid retrieval: fuse vector-cosine and BM25 rankings via RRF."""
        pool = self.facts_in(category) or self.facts
        if not pool:
            return []

        q = np.array(llm.embed(query), dtype=float)
        qn = np.linalg.norm(q) or 1.0

        def cos(vec):
            a = np.array(vec, dtype=float)
            return float(np.dot(q, a) / (qn * (np.linalg.norm(a) or 1.0)))

        vec_ranked = sorted(pool, key=lambda x: cos(x["embedding"]), reverse=True)

        bm = BM25Okapi([_tok(x["text"]) for x in pool])
        scores = bm.get_scores(_tok(query))
        bm_ranked = [pool[i] for i in np.argsort(scores)[::-1]]

        # Reciprocal Rank Fusion
        rr = {}
        for rank, x in enumerate(vec_ranked):
            rr[x["id"]] = rr.get(x["id"], 0.0) + 1.0 / (60 + rank)
        for rank, x in enumerate(bm_ranked):
            rr[x["id"]] = rr.get(x["id"], 0.0) + 1.0 / (60 + rank)

        fused = sorted(pool, key=lambda x: rr.get(x["id"], 0.0), reverse=True)
        return fused[:k]


store = Store()
