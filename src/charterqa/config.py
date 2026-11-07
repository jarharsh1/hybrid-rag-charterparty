"""Runtime configuration (env with local defaults). Ollama-first, offline fallback."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3.2")
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
    offline: bool = os.getenv("RAG_OFFLINE", "0") == "1"
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    # How many candidates each retriever contributes before fusion.
    candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "10"))
    # Reciprocal-rank-fusion constant.
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))


settings = Settings()
