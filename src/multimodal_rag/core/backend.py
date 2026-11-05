"""Compatibility exports for the Streamlit application."""

from .config import TOP_K, RELEVANCE_THRESHOLD
from .embeddings import MiniLMEmbeddings
from .llm import generate_ollama_response
from .sources import get_source_info
from .prompts import build_prompt
from .rag_engine import ask_rag, vectorstore

__all__ = [
    "TOP_K",
    "RELEVANCE_THRESHOLD",
    "MiniLMEmbeddings",
    "generate_ollama_response",
    "get_source_info",
    "build_prompt",
    "ask_rag",
    "vectorstore",
]
