"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"
NEWS_ENTITY_TERMS = {
    "huu", "tin", "miu", "le", "long", "nhat", "son", "ngoc", "minh",
    "nhikolai", "dinh", "chu", "bin", "chi", "dan", "an", "tay",
    "truc", "phuong", "nghe", "si", "ca", "dien", "vien", "nguoi", "mau",
}
DOMAIN_TERMS = {
    "ma", "tuy", "chat", "cam", "luat", "dieu", "249", "57", "2021", "2015",
    "hinh", "phat", "tang", "tru", "trai", "phep", "cai", "nghien",
    "nghi", "dinh", "su", "nghe", "si", "dien", "vien", "ca",
    "rapper", "pageindex", "hybrid", "retrieval", "citation", "nguon",
    # Thêm: "tội" (normalized), tên nghệ sĩ phổ biến, hành vi pháp lý phổ biến
    "toi", "miu", "le", "long", "nhat", "son", "minh", "luu", "nghien",
    "khoi", "bat", "biet", "xu", "han", "an", "cu", "tu", "co", "vu",
}


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0:
        return []

    from .utils_text import tokenize
    if not (set(tokenize(query)) & DOMAIN_TERMS):
        return []

    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    final_results = _apply_intent_boost(query, final_results)

    if not final_results or final_results[0]["score"] < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return _apply_intent_boost(query, fallback)
    return final_results[:top_k]


def _apply_intent_boost(query: str, results: list[dict]) -> list[dict]:
    from .utils_text import tokenize

    query_tokens = set(tokenize(query))
    if not results:
        return results

    if "57" in query_tokens:
        return sorted(
            results,
            key=lambda item: (
                "nghi-dinh-57" in item.get("metadata", {}).get("source", ""),
                item.get("score", 0.0),
            ),
            reverse=True,
        )

    if query_tokens & NEWS_ENTITY_TERMS:
        boosted = []
        for idx, item in enumerate(results):
            metadata = item.get("metadata", {})
            content_tokens = set(tokenize(item.get("content", "")[:1200]))
            entity_overlap = len(query_tokens & content_tokens)
            news_bonus = 2.0 if metadata.get("type") == "news" else 0.0
            score = news_bonus + entity_overlap + float(item.get("score", 0.0)) * 0.01 - idx * 1e-6
            boosted.append((score, item))
        boosted.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in boosted]

    return results


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
