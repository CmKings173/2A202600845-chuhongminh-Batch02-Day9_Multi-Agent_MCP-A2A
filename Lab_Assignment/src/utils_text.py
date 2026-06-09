"""Shared lightweight text utilities for the local RAG implementation."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

from . import config

VECTOR_DIM = 256


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def text_to_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dim
    for token, count in counts.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        vector[idx] += float(count)
    norm = math.sqrt(sum(v * v for v in vector))
    if norm:
        vector = [v / norm for v in vector]
    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts according to .env config, with deterministic local fallback."""
    if config.EMBEDDING_PROVIDER == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.EMBEDDING_MODEL)
            return [embedding.tolist() for embedding in model.encode(texts, show_progress_bar=False)]
        except Exception:
            if not config.ALLOW_LOCAL_FALLBACK:
                raise
    return [text_to_vector(text) for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    t_tokens = set(tokenize(text))
    return len(q_tokens & t_tokens) / len(q_tokens)


def ensure_standardized_data() -> None:
    from .task3_convert_markdown import convert_all, OUTPUT_DIR

    if not OUTPUT_DIR.exists() or not list(OUTPUT_DIR.rglob("*.md")):
        convert_all()


def load_or_build_chunks() -> list[dict]:
    import json
    from .task4_chunking_indexing import INDEX_PATH, load_documents, chunk_documents, embed_chunks, index_to_vectorstore

    ensure_standardized_data()
    if INDEX_PATH.exists():
        try:
            chunks = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if chunks:
                return chunks
        except Exception:
            pass
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_to_vectorstore(chunks)
    return chunks
