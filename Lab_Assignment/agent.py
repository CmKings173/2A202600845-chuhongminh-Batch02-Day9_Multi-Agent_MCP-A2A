"""
Lab_Assignment/agent.py — Supervisor-Workers Multi-Agent RAG System
====================================================================

Kiến trúc: Supervisor - Workers pattern sử dụng LangGraph.

Supervisor quyết định gọi Worker nào dựa vào câu hỏi người dùng:
  - legal_worker:     Tìm kiếm văn bản pháp luật (drug law documents)
  - news_worker:      Tìm kiếm bài báo nghệ sĩ (celebrity news)
  - pageindex_worker: Fallback vectorless search (PageIndex API)

Sau khi thu thập đủ context, Supervisor chuyển sang FINISH để
tổng hợp câu trả lời cuối cùng có kèm trích dẫn (citations).
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# ── Load environment ─────────────────────────────────────────────────────────
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_AGENT_DIR, ".env"))

# Make local src importable
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)


# ── LLM factory ──────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return the configured LLM (Ollama / OpenAI / Gemini)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
        )
    if provider in ("gemini", "google"):
        return ChatOpenAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            openai_api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=temperature,
        )
    # Default: local Ollama
    return ChatOpenAI(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        openai_api_key="ollama",
        base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        temperature=temperature,
    )


# ── State definition ─────────────────────────────────────────────────────────

def _concat(a: str, b: str) -> str:
    """Reducer: nối thêm kết quả mới vào context hiện tại."""
    return (a + "\n\n" + b).strip() if b else a


WORKER_NAMES = Literal["legal_worker", "news_worker", "pageindex_worker", "FINISH"]


class AgentState(TypedDict):
    question: str                          # Câu hỏi gốc của người dùng
    context: Annotated[str, _concat]       # Context tổng hợp từ các Workers
    next: str                              # Worker tiếp theo Supervisor quyết định gọi
    iterations: int                        # Số vòng lặp đã thực hiện
    final_answer: str                      # Câu trả lời cuối cùng có citation


# ── Supervisor Node ───────────────────────────────────────────────────────────

_SUPERVISOR_SYSTEM = """You are a legal research coordinator. Your job is to decide which specialist worker to call next.

Available workers:
- legal_worker: Search drug law documents (Vietnamese drug law, criminal code, etc.)
- news_worker: Search celebrity news about drug-related arrests and incidents
- pageindex_worker: Vectorless fallback search (use only when other workers returned insufficient results)
- FINISH: All necessary information gathered, generate the final answer

Rules:
1. Always call legal_worker first if the question involves law, penalties, or regulations.
2. Call news_worker if the question mentions celebrities, artists, or specific incidents.
3. Call pageindex_worker only if context is still insufficient after 2+ rounds.
4. Call FINISH when you have enough context to answer fully, or after 3 rounds.
5. Never repeat the same worker twice.

Current context summary (may be empty initially):
{context_summary}

Workers already called: {workers_called}

Respond with ONLY one of: legal_worker | news_worker | pageindex_worker | FINISH
"""


def supervisor_node(state: AgentState) -> dict:
    """Supervisor: quyết định worker tiếp theo cần gọi."""
    llm = _get_llm()
    context_summary = state.get("context", "")[:500] or "(empty)"
    iterations = state.get("iterations", 0)

    # Giới hạn vòng lặp để tránh vòng lặp vô tận
    if iterations >= 3:
        return {"next": "FINISH", "iterations": iterations + 1}

    # Đếm workers đã gọi từ context để tránh gọi lại
    ctx = state.get("context", "")
    workers_called = []
    if "## Legal Search Results" in ctx:
        workers_called.append("legal_worker")
    if "## News Search Results" in ctx:
        workers_called.append("news_worker")
    if "## PageIndex Results" in ctx:
        workers_called.append("pageindex_worker")

    messages = [
        SystemMessage(content=_SUPERVISOR_SYSTEM.format(
            context_summary=context_summary,
            workers_called=workers_called or "none",
        )),
        HumanMessage(content=f"Question: {state['question']}\n\nWhich worker should be called next?"),
    ]

    response = llm.invoke(messages)
    decision = response.content.strip().lower()

    # Xác nhận decision hợp lệ
    valid = {"legal_worker", "news_worker", "pageindex_worker", "finish"}
    for choice in ["legal_worker", "news_worker", "pageindex_worker", "FINISH"]:
        if choice.lower() in decision:
            return {"next": choice, "iterations": iterations + 1}

    # Mặc định: nếu không rõ ràng thì FINISH
    return {"next": "FINISH", "iterations": iterations + 1}


# ── Worker Nodes ──────────────────────────────────────────────────────────────

def legal_worker_node(state: AgentState) -> dict:
    """Worker 1: Tìm kiếm văn bản pháp luật về ma tuý."""
    try:
        from src.task9_retrieval_pipeline import retrieve
        results = retrieve(state["question"], top_k=5)
    except Exception as exc:
        return {"context": f"## Legal Search Results\n[Lỗi kết nối RAG: {exc}]"}

    if not results:
        return {"context": "## Legal Search Results\n[Không tìm thấy văn bản pháp luật liên quan.]"}

    formatted = "\n\n".join(
        f"[{i+1}] (score={r['score']:.3f}, source={r['metadata'].get('source', '?')})\n{r['content'][:600]}"
        for i, r in enumerate(results)
    )
    return {"context": f"## Legal Search Results\n{formatted}"}


def news_worker_node(state: AgentState) -> dict:
    """Worker 2: Tìm kiếm bài báo về nghệ sĩ liên quan ma tuý."""
    try:
        from src.task9_retrieval_pipeline import retrieve
        # Thêm từ khoá để tập trung vào news
        news_query = state["question"] + " nghệ sĩ diễn viên ca sĩ"
        results = retrieve(news_query, top_k=5)
        # Ưu tiên loại news
        results = sorted(results, key=lambda r: (r.get("metadata", {}).get("type") == "news"), reverse=True)
    except Exception as exc:
        return {"context": f"## News Search Results\n[Lỗi kết nối RAG: {exc}]"}

    if not results:
        return {"context": "## News Search Results\n[Không tìm thấy bài báo liên quan.]"}

    formatted = "\n\n".join(
        f"[{i+1}] (score={r['score']:.3f}, type={r['metadata'].get('type', '?')})\n{r['content'][:600]}"
        for i, r in enumerate(results)
    )
    return {"context": f"## News Search Results\n{formatted}"}


def pageindex_worker_node(state: AgentState) -> dict:
    """Worker 3: Tìm kiếm vectorless qua PageIndex (fallback)."""
    try:
        from src.task8_pageindex_vectorless import pageindex_search
        results = pageindex_search(state["question"], top_k=5)
    except Exception as exc:
        return {"context": f"## PageIndex Results\n[Lỗi PageIndex: {exc}]"}

    if not results:
        return {"context": "## PageIndex Results\n[Không tìm thấy kết quả từ PageIndex.]"}

    formatted = "\n\n".join(
        f"[{i+1}] (score={r['score']:.3f})\n{r['content'][:600]}"
        for i, r in enumerate(results)
    )
    return {"context": f"## PageIndex Results\n{formatted}"}


# ── Synthesizer Node ──────────────────────────────────────────────────────────

_SYNTHESIZER_SYSTEM = """You are a senior legal analyst. Based on the research context below, answer the user's question comprehensively.

Rules:
- Cite your sources using format [Nguồn: <source name>]
- If you refer to a specific law or article, cite it explicitly.
- If the context lacks information, state "Tôi không thể xác minh điều này" instead of guessing.
- Write in Vietnamese.
- Structure your answer with clear sections if the question is complex.

Research Context:
{context}
"""


def synthesizer_node(state: AgentState) -> dict:
    """Synthesizer: tổng hợp câu trả lời cuối cùng có kèm trích dẫn."""
    llm = _get_llm(temperature=0.2)
    context = state.get("context", "(không có dữ liệu)")

    messages = [
        SystemMessage(content=_SYNTHESIZER_SYSTEM.format(context=context)),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)
    return {"final_answer": response.content}


# ── Routing Function ─────────────────────────────────────────────────────────

def route_from_supervisor(state: AgentState) -> str:
    """Hàm định tuyến: đọc state["next"] và trả về tên node tiếp theo."""
    return state.get("next", "FINISH")


# ── Graph Construction ────────────────────────────────────────────────────────

def build_supervisor_graph() -> StateGraph:
    """Xây dựng và compile LangGraph Supervisor-Workers graph."""
    graph = StateGraph(AgentState)

    # Đăng ký các nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("legal_worker", legal_worker_node)
    graph.add_node("news_worker", news_worker_node)
    graph.add_node("pageindex_worker", pageindex_worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point: bắt đầu từ Supervisor
    graph.add_edge(START, "supervisor")

    # Conditional edges từ Supervisor → Workers hoặc Synthesizer
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "legal_worker": "legal_worker",
            "news_worker": "news_worker",
            "pageindex_worker": "pageindex_worker",
            "FINISH": "synthesizer",
        },
    )

    # Sau mỗi Worker → quay lại Supervisor để quyết định tiếp
    graph.add_edge("legal_worker", "supervisor")
    graph.add_edge("news_worker", "supervisor")
    graph.add_edge("pageindex_worker", "supervisor")

    # Synthesizer → END
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(question: str) -> dict:
    """
    Chạy Supervisor-Workers agent với câu hỏi đầu vào.

    Args:
        question: Câu hỏi về pháp luật ma tuý hoặc nghệ sĩ liên quan.

    Returns:
        dict với các keys:
            - 'final_answer': Câu trả lời có citation
            - 'context': Toàn bộ context thu thập được
            - 'iterations': Số vòng Supervisor thực hiện
    """
    graph = build_supervisor_graph()
    result = graph.invoke({
        "question": question,
        "context": "",
        "next": "",
        "iterations": 0,
        "final_answer": "",
    })
    return {
        "final_answer": result.get("final_answer", ""),
        "context": result.get("context", ""),
        "iterations": result.get("iterations", 0),
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Hình phạt cho tội tàng trữ ma tuý là gì?"
    print(f"\nCâu hỏi: {q}")
    print("=" * 70)
    result = run_agent(q)
    print("\n📋 KẾT QUẢ:")
    print(result["final_answer"])
    print(f"\n[Iterations: {result['iterations']}]")
