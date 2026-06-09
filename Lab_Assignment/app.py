"""
Lab_Assignment/app.py — Drug Law RAG Chatbot
============================================

Giao diện Streamlit kết nối với Supervisor-Workers Agent.

Chạy: streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Law RAG Chatbot | Supervisor-Workers",
    page_icon="⚖️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .main {
    font-family: 'Outfit', sans-serif !important;
    background: #0f1117;
    color: #e2e8f0;
}

.main-title {
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 60%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.6rem;
    margin-bottom: 0.15rem;
    margin-top: -1.5rem;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}

.arch-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #a5b4fc;
    margin-bottom: 1.2rem;
}

.source-card {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #818cf8;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.7rem;
    border-radius: 4px;
}

.source-label {
    color: #818cf8;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
}

.source-content {
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.4;
    white-space: pre-wrap;
}

.worker-tag {
    display: inline-block;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #34d399;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">⚖️ Drug Law RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Hệ thống hỏi đáp thông minh về Luật phòng chống ma túy & sự kiện nghệ sĩ</p>', unsafe_allow_html=True)
st.markdown('<span class="arch-badge">🤖 Supervisor-Workers Pattern (Day 09 Assignment)</span>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Kiến trúc hệ thống")
    st.markdown("""
```
User Question
    ↓
  Supervisor (LLM)
    ↓
  ┌──────────────────┐
  │  legal_worker    │ → Pháp luật ma tuý
  │  news_worker     │ → Bài báo nghệ sĩ
  │  pageindex_worker│ → Fallback search
  └──────────────────┘
    ↓
  Synthesizer (LLM)
    ↓
Final Answer + Citations
```
    """)
    st.markdown("---")
    st.markdown("### 💡 Câu hỏi ví dụ")
    st.markdown("""
- Hình phạt tàng trữ ma tuý là gì?
- Nghệ sĩ nào bị bắt vì ma tuý?
- Điều 249 BLHS quy định gì?
- Quy trình cai nghiện bắt buộc theo luật 2021?
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Cấu hình")
    show_context = st.checkbox("Hiển thị context thu thập được", value=False)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("workers"):
            workers_html = " ".join(
                f'<span class="worker-tag">{w}</span>'
                for w in msg["workers"]
            )
            st.markdown(f"Workers: {workers_html}", unsafe_allow_html=True)
        if show_context and msg.get("context"):
            with st.expander("📚 Context thu thập được"):
                st.text(msg["context"][:3000])

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Nhập câu hỏi về pháp luật ma tuý hoặc tin tức liên quan...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ Đang tìm kiếm và phân tích...")

        start = time.time()
        from agent import run_agent
        result = run_agent(prompt)
        elapsed = time.time() - start

        status_placeholder.empty()

        # Hiển thị câu trả lời
        st.markdown(result["final_answer"])

        # Hiển thị metadata
        ctx = result.get("context", "")
        workers = []
        if "## Legal Search Results" in ctx:
            workers.append("⚖️ legal_worker")
        if "## News Search Results" in ctx:
            workers.append("📰 news_worker")
        if "## PageIndex Results" in ctx:
            workers.append("🔍 pageindex_worker")

        if workers:
            workers_html = " ".join(f'<span class="worker-tag">{w}</span>' for w in workers)
            st.markdown(f"Workers gọi: {workers_html} &nbsp;|&nbsp; ⏱ {elapsed:.1f}s &nbsp;|&nbsp; 🔄 {result.get('iterations', 0)} vòng",
                        unsafe_allow_html=True)

        if show_context and ctx:
            with st.expander("📚 Context thu thập được"):
                st.text(ctx[:3000])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_answer"],
        "workers": workers,
        "context": result.get("context", ""),
    })
