"""Central config for the on-device AI prototype."""
import os

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
DATA_FILE = os.path.join(DATA_DIR, "memories.json")

# Local Ollama models (already pulled on this machine).
# 1.5B chat model: fits a 4GB GPU fully (no VRAM swapping) -> much faster,
# same pipeline. Override with CHAT_MODEL=llama3.2:3b for higher quality.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3:1.7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Seeded "life bucket" expert memories.
CATEGORIES = ["Health", "Money", "Work", "Learning", "Home", "Personal", "Travel"]

SEARCH_K = 8   # candidates pulled by hybrid search
TOP_K = 5      # notes actually fed to the LLM after fusion

# If the best-matching note's cosine similarity is below this, the message isn't
# a memory question -> answer as a normal chat instead of invoking RAG.
# Calibrated: memory Qs score ~0.54-0.78 ("what is my name?" vs a short name
# fact = 0.54, the observed floor), general Qs ~0.38-0.46 ("tell me a joke" =
# 0.46, the observed ceiling). 0.52 splits the measured gap.
RAG_THRESHOLD = float(os.environ.get("RAG_THRESHOLD", "0.52"))

# Context window cap — keeps the KV cache small enough for this machine.
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))
