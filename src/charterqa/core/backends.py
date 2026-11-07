"""Pluggable embedding + LLM backends (Ollama-first, deterministic offline fallback).

Mirrors the design used across this maritime RAG set: Ollama when reachable,
otherwise a hashed bag-of-words embedder and an extractive LLM so the pipeline
and tests run with no model server.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from charterqa.config import settings

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class HashingEmbedder:
    dim = _EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return vec if norm == 0 else [v / norm for v in vec]


class ExtractiveLLM:
    def generate(self, prompt: str) -> str:
        context, question = _split_prompt(prompt)
        sentences = _sentences(context)
        if not sentences:
            return "I don't have enough context to answer that."
        q = set(tokenize(question))
        scored = sorted(sentences, key=lambda s: len(q & set(tokenize(s))), reverse=True)
        top = [s for s in scored[:2] if q & set(tokenize(s))]
        return " ".join(top) if top else "The contract text does not appear to cover that."


def _split_prompt(prompt: str) -> tuple[str, str]:
    context, question = "", prompt
    if "CONTEXT:" in prompt and "QUESTION:" in prompt:
        context = prompt.split("CONTEXT:", 1)[1].split("QUESTION:", 1)[0].strip()
        question = prompt.split("QUESTION:", 1)[1].strip()
    return context, question


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host, self.model = host.rstrip("/"), model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        with httpx.Client(timeout=60) as c:
            for t in texts:
                r = c.post(f"{self.host}/api/embeddings", json={"model": self.model, "prompt": t})
                r.raise_for_status()
                out.append(r.json()["embedding"])
        return out


class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host, self.model = host.rstrip("/"), model

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=120) as c:
            r = c.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            return r.json()["response"].strip()


def _ollama_up(host: str) -> bool:
    try:
        with httpx.Client(timeout=2) as c:
            return c.get(f"{host.rstrip('/')}/api/tags").status_code == 200
    except Exception:
        return False


def get_backends() -> tuple[Embedder, LLM]:
    if not settings.offline and _ollama_up(settings.ollama_host):
        return (
            OllamaEmbedder(settings.ollama_host, settings.embed_model),
            OllamaLLM(settings.ollama_host, settings.llm_model),
        )
    return HashingEmbedder(), ExtractiveLLM()
