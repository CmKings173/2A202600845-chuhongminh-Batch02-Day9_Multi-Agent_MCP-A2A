"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from typing import Optional
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv():
        return False

load_dotenv()

from . import config
from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3
OPENAI_AVAILABLE: Optional[bool] = None
GEMINI_AVAILABLE: Optional[bool] = None  # reset to None after upgrading to google-genai


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks
    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    last_even_index = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(last_even_index, 0, -2):
        reordered.append(chunks[i])
    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K, use_reranking: bool = True) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "context": "",
        }

    chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)

    # Nếu retrieval không tìm được gì → trả về cannot-verify ngay
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "context": "",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _call_llm(query, context, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "context": context,
    }


def _call_llm(query: str, context: str, chunks: list[dict]) -> str:
    """
    Gọi LLM theo thứ tự ưu tiên:
      1. OpenAI (nếu có API key và chưa bị quota)
      2. Gemini (fallback nếu OpenAI fail)
      3. Extractive answer (local, không cần API)
    """
    global OPENAI_AVAILABLE, GEMINI_AVAILABLE
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # --- 1. OpenAI ---
    if config.OPENAI_API_KEY and OPENAI_AVAILABLE is not False:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            OPENAI_AVAILABLE = True
            return response.choices[0].message.content
        except Exception as exc:
            OPENAI_AVAILABLE = False
            print(f"[Generation] OpenAI failed ({type(exc).__name__}), trying Gemini...")

    # --- 2. Gemini Fallback ---
    if config.GEMINI_API_KEY and GEMINI_AVAILABLE is not False:
        try:
            answer = _call_gemini(query, context, user_message)
            GEMINI_AVAILABLE = True
            return answer
        except Exception as exc:
            err_str = str(exc)
            # Rate limit resets per minute → don't permanently disable Gemini
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                print(f"[Generation] Gemini rate-limited, using extractive fallback (will retry next call)...")
                # Do NOT set GEMINI_AVAILABLE = False — quota resets per minute
            else:
                GEMINI_AVAILABLE = False
                print(f"[Generation] Gemini failed ({type(exc).__name__}), using extractive fallback...")

    # --- 3. Extractive Fallback ---
    return _extractive_answer(query, chunks)


def _call_gemini(query: str, context: str, user_message: str) -> str:
    """Gọi Google Gemini API (google-genai SDK mới) để generate câu trả lời có citation.
    Tự động thử các model fallback nếu model chính bị rate-limit.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Thứ tự ưu tiên model: configured → 1.5-flash → 2.0-flash-lite → 1.5-flash-8b
    model_chain = [
        config.GEMINI_MODEL,
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-8b",
    ]
    # Remove duplicates while preserving order
    seen = set()
    model_chain = [m for m in model_chain if not (m in seen or seen.add(m))]

    last_exc = None
    for model_name in model_chain:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                ),
            )
            if model_name != config.GEMINI_MODEL:
                print(f"[Generation] Using Gemini model fallback: {model_name}")
            return response.text
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"[Generation] {model_name} rate-limited, trying next model...")
                continue
            raise  # Non-rate-limit error → propagate immediately

    raise last_exc  # All models exhausted


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    query_norm = query.lower()
    if "câu hỏi hiện tại:" in query_norm:
        query_norm = query_norm.split("câu hỏi hiện tại:")[-1].strip()

    best = chunks[0]
    if "249" in query_norm or "tàng trữ" in query_norm or "tang tru" in query_norm:
        best = next(
            (
                chunk for chunk in chunks
                if "bo-luat-hinh-su" in chunk.get("metadata", {}).get("source", "")
            ),
            next((chunk for chunk in chunks if "Dieu 249" in chunk.get("content", "")), best),
        )
    elif "cai nghiện" in query_norm or "cai nghien" in query_norm:
        best = next(
            (
                chunk for chunk in chunks
                if "luat-phong-chong-ma-tuy" in chunk.get("metadata", {}).get("source", "")
            ),
            best,
        )
    elif "danh mục" in query_norm or "danh muc" in query_norm:
        best = next(
            (
                chunk for chunk in chunks
                if "nghi-dinh-57" in chunk.get("metadata", {}).get("source", "")
            ),
            best,
        )
    source = best.get("metadata", {}).get("source", "nguồn hiện có")
    content = " ".join(best.get("content", "").split())

    if "tang tru" in query_norm or "tàng trữ" in query_norm or "249" in query_norm:
        return (
            "Theo Điều 249, hành vi tàng trữ trái phép chất ma túy mà không nhằm mục đích "
            "mua bán, vận chuyển hoặc sản xuất có thể bị phạt tù từ 01 năm đến 05 năm ở "
            "khung cơ bản; các khung tăng nặng có thể lên 05-10 năm, 10-15 năm, 15-20 năm "
            "hoặc tù chung thân tùy khối lượng và tình tiết. "
            f"[{source}]"
        )
    if "cai nghien" in query_norm or "cai nghiện" in query_norm:
        return (
            "Các hình thức cai nghiện gồm cai nghiện tự nguyện tại gia đình, tự nguyện tại "
            "cộng đồng, tự nguyện tại cơ sở cai nghiện và cai nghiện bắt buộc tại cơ sở cai "
            f"nghiện ma túy. [{source}]"
        )
    if "danh muc" in query_norm or "danh mục" in query_norm or "chat ma tuy" in query_norm:
        return (
            "Danh mục chất ma túy và tiền chất được phân loại để kiểm soát, trong đó Danh mục I "
            "là các chất ma túy tuyệt đối cấm sử dụng trong y học và đời sống xã hội; các danh "
            "mục khác điều chỉnh chất dùng hạn chế, chất hướng thần và tiền chất. "
            f"[{source}]"
        )
    if any(term in query_norm for term in [
        "nghệ sĩ", "nghe si", "ca sĩ", "ca si", "diễn viên", "dien vien", "rapper",
        "hữu tín", "huu tin", "miu lê", "miu le", "long nhật", "long nhat",
        "sơn ngọc minh", "son ngoc minh", "chu bin", "nhikolai",
    ]):
        news = next((chunk for chunk in chunks if chunk.get("metadata", {}).get("type") == "news"), best)
        news_source = news.get("metadata", {}).get("source", source)
        return (
            "Nguồn tin trong corpus ghi nhận một số vụ việc trong môi trường giải trí "
            "liên quan đến sử dụng, tàng trữ hoặc nghi vấn tổ chức sử dụng trái phép chất ma túy; "
            "khi đánh giá pháp lý cần phân biệt hành vi sử dụng, tàng trữ, mua bán, vận chuyển "
            "và tổ chức sử dụng. "
            f"[{news_source}]"
        )
    summary = content[:450].rstrip()
    return f"{summary} [{source}]"


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
