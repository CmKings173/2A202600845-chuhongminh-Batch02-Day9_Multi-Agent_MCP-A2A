"""Central runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional for tests
    def load_dotenv():
        return False


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_plain_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_plain_env_file()

VECTOR_STORE = os.getenv("VECTOR_STORE", "local").lower()
LOCAL_INDEX_PATH = Path(os.getenv("LOCAL_INDEX_PATH", DATA_DIR / "local_index.json"))

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
if WEAVIATE_URL and not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK", "true").lower() in {"1", "true", "yes"}

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local_hash").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
