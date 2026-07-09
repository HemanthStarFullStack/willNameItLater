"""Central config for the on-device AI prototype."""
import os

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
DATA_FILE = os.path.join(DATA_DIR, "memories.json")

# Local Ollama models (already pulled on this machine).
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Seeded "life bucket" expert memories.
CATEGORIES = ["Health", "Money", "Work", "Learning", "Home", "Personal", "Travel"]

SEARCH_K = 8   # candidates pulled by hybrid search
TOP_K = 5      # notes actually fed to the LLM after fusion

# Context window cap — keeps the KV cache small enough for this machine.
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))
