"""HHEM groundedness check — the "never lies" layer.

HHEM (Vectara's Hughes Hallucination Evaluation Model, HHEM-2.1-Open) is a
small cross-encoder: it takes (evidence, claim) as ONE input and outputs the
probability 0..1 that the claim is actually supported by the evidence. It's
NLI (entailment) specialized for RAG — unlike lexical overlap it understands
paraphrase AND catches negation/wrong-value ("allergic to penicillin" vs
"not allergic to penicillin" overlaps 100% lexically but scores ~0).

~440MB, flan-t5-base backbone, runs on CPU in well under a second per pair.
Loaded lazily; cached under HF_HOME (the data volume) so rebuilds don't
re-download it.
"""
import threading

_model = None
_lock = threading.Lock()


def _load():
    global _model
    with _lock:
        if _model is None:
            from transformers import AutoModelForSequenceClassification
            _model = AutoModelForSequenceClassification.from_pretrained(
                "vectara/hallucination_evaluation_model",
                trust_remote_code=True)
            _model.eval()
    return _model


def warmup():
    """Load the model off the critical path (call from a startup thread)."""
    try:
        _load()
        return True
    except Exception:
        return False


def score(evidence, claim):
    """P(claim is supported by evidence). Raises if transformers/torch are
    missing — callers fall back to the lexical + LLM-judge path."""
    return float(_load().predict([(evidence, claim)])[0])
