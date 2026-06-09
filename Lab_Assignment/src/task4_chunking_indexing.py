"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import json
import requests

from . import config

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_PATH = config.LOCAL_INDEX_PATH


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive chunking an toàn cho legal/news markdown không đồng nhất heading.
CHUNK_SIZE = 500        # Đủ ngắn để citation rõ, đủ dài để giữ ngữ cảnh điều luật.
CHUNK_OVERLAP = 50      # Giữ nối cảnh giữa hai chunk liền kề, nhưng không lặp quá nhiều.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn BAAI/bge-m3 cho production vì multilingual, phù hợp tiếng Việt.
# Bản local_hash trong repo giúp chạy test/demo offline khi chưa có model/API.
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIM = config.EMBEDDING_DIM

VECTOR_STORE = config.VECTOR_STORE  # "local" | "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc in documents:
        text = doc["content"]
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                window = text[start:end]
                split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
                if split_at > CHUNK_SIZE * 0.5:
                    end = start + split_at + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": chunk_index},
                })
                chunk_index += 1
            if end >= len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from .utils_text import embed_texts

    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    if VECTOR_STORE == "weaviate":
        try:
            return _index_to_weaviate(chunks)
        except Exception as exc:
            if not config.ALLOW_LOCAL_FALLBACK:
                raise
            print(f"⚠ Weaviate indexing failed, fallback local JSON: {exc}")

    return INDEX_PATH


def _index_to_weaviate(chunks: list[dict]):
    """Index chunks to Weaviate when VECTOR_STORE=weaviate is configured."""
    if not config.WEAVIATE_URL:
        raise RuntimeError("WEAVIATE_URL is required when VECTOR_STORE=weaviate")

    import weaviate
    from weaviate.classes.config import DataType, Property
    from weaviate.classes.init import Auth

    auth = Auth.api_key(config.WEAVIATE_API_KEY) if config.WEAVIATE_API_KEY else None
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=auth,
    )
    try:
        if client.collections.exists(config.WEAVIATE_COLLECTION):
            client.collections.delete(config.WEAVIATE_COLLECTION)
        try:
            collection = client.collections.create(
                name=config.WEAVIATE_COLLECTION,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )
        except Exception as exc:
            if "hfresh" not in str(exc).lower():
                raise
            _create_weaviate_collection_raw()
            collection = client.collections.get(config.WEAVIATE_COLLECTION)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                    },
                    vector=chunk.get("embedding"),
                )
        return config.WEAVIATE_COLLECTION
    finally:
        client.close()


def _create_weaviate_collection_raw() -> None:
    """Create collection via REST for Weaviate clusters that require hfresh."""
    base_url = config.WEAVIATE_URL.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if config.WEAVIATE_API_KEY:
        headers["Authorization"] = f"Bearer {config.WEAVIATE_API_KEY}"

    schema = {
        "class": config.WEAVIATE_COLLECTION,
        "vectorizer": "none",
        "vectorIndexType": "hfresh",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "doc_type", "dataType": ["text"]},
            {"name": "chunk_index", "dataType": ["int"]},
        ],
    }
    response = requests.post(f"{base_url}/v1/schema", headers=headers, json=schema, timeout=30)
    if response.status_code not in {200, 201}:
        raise RuntimeError(f"Raw Weaviate schema create failed: HTTP {response.status_code} {response.text[:300]}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
