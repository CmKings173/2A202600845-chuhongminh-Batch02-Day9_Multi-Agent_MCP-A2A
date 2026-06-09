"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math
from collections import Counter

from .utils_text import load_or_build_chunks, tokenize

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized = [tokenize(doc["content"]) for doc in corpus]
    doc_freq: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avgdl": avgdl, "n": len(tokenized)}


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0:
        return []

    global CORPUS
    if not CORPUS:
        CORPUS = [
            {"content": c["content"], "metadata": c.get("metadata", {})}
            for c in load_or_build_chunks()
        ]
    if not CORPUS:
        return []

    index = build_bm25_index(CORPUS)
    query_tokens = tokenize(query)
    k1 = 1.5
    b = 0.75
    results = []
    for idx, tokens in enumerate(index["tokenized"]):
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            df = index["doc_freq"].get(token, 0)
            idf = math.log(1 + (index["n"] - df + 0.5) / (df + 0.5))
            numerator = tf[token] * (k1 + 1)
            denominator = tf[token] + k1 * (1 - b + b * dl / max(index["avgdl"], 1))
            score += idf * numerator / denominator
        if score > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(score),
                "metadata": CORPUS[idx]["metadata"],
            })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
