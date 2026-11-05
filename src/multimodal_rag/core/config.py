import os
from pathlib import Path

# Project root: AetherMind_AI
PROJECT_DIR = Path(__file__).resolve().parents[3]

# Persistent FAISS index
INDEX_DIR = PROJECT_DIR / "indexes" / "multimodal_rag_faiss"

# Ollama configuration
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL_NAME = "qwen2.5vl:3b"

# Retrieval configuration
TOP_K = 8
RELEVANCE_THRESHOLD = 1.00
SHOW_EVIDENCE = True
