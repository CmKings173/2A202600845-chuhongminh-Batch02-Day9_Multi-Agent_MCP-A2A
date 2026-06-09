"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional

from . import config
from .utils_text import keyword_overlap_score


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or not candidates or top_k <= 0:
        return []

    if config.JINA_API_KEY and candidates:
        try:
            return _rerank_jina_api(query, candidates, top_k)
        except Exception:
            if not config.ALLOW_LOCAL_FALLBACK:
                raise

    reranked = []
    for rank, candidate in enumerate(candidates, 1):
        base_score = float(candidate.get("score", 0.0))
        overlap = keyword_overlap_score(query, candidate.get("content", ""))
        score = 0.65 * overlap + 0.35 * base_score + 1e-6 / rank
        item = candidate.copy()
        item["score"] = float(score)
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def _rerank_jina_api(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {config.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.JINA_RERANK_MODEL,
            "query": query,
            "documents": [candidate.get("content", "") for candidate in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = []
    for result in payload.get("results", []):
        item = candidates[result["index"]].copy()
        item["score"] = float(result.get("relevance_score", result.get("score", 0.0)))
        results.append(item)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates or top_k <= 0:
        return []

    from .utils_text import cosine_similarity, text_to_vector

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    vectors = [c.get("embedding") or text_to_vector(c.get("content", "")) for c in candidates]
    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = cosine_similarity(query_embedding, vectors[idx])
            diversity_penalty = max(
                [cosine_similarity(vectors[idx], vectors[sel_idx]) for sel_idx in selected] or [0.0]
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [{**candidates[i], "score": float(candidates[i].get("score", 0.0))} for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists or top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or not candidates or top_k <= 0:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from .utils_text import text_to_vector
        return rerank_mmr(text_to_vector(query), candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
