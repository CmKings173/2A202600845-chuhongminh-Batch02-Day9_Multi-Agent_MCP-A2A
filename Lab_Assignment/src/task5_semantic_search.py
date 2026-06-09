"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0:
        return []

    from . import config
    from .utils_text import cosine_similarity, embed_texts, load_or_build_chunks, text_to_vector

    if config.VECTOR_STORE == "weaviate" and config.WEAVIATE_URL:
        try:
            return _semantic_search_weaviate(query, top_k)
        except Exception:
            if not config.ALLOW_LOCAL_FALLBACK:
                raise

    query_embedding = embed_texts([query])[0]
    scored = []
    for chunk in load_or_build_chunks():
        embedding = chunk.get("embedding") or text_to_vector(chunk["content"])
        score = cosine_similarity(query_embedding, embedding)
        if score > 0:
            scored.append({
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _semantic_search_weaviate(query: str, top_k: int) -> list[dict]:
    from . import config
    from .utils_text import embed_texts

    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery

    auth = Auth.api_key(config.WEAVIATE_API_KEY) if config.WEAVIATE_API_KEY else None
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=auth,
    )
    try:
        collection = client.collections.get(config.WEAVIATE_COLLECTION)
        response = collection.query.near_vector(
            near_vector=embed_texts([query])[0],
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        results = []
        for obj in response.objects:
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0
            results.append({
                "content": props.get("content", ""),
                "score": float(1 - distance),
                "metadata": {
                    "source": props.get("source", ""),
                    "type": props.get("doc_type", ""),
                    "chunk_index": props.get("chunk_index", 0),
                },
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results
    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
