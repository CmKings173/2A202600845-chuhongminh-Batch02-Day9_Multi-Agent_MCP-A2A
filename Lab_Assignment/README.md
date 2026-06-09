# Lab_Assignment — Supervisor-Workers RAG Agent

**Day 09 — Bài Tập Assignment | Batch02 | Chuhongminh**

> Nâng cấp RAG Pipeline từ Day 08 thành hệ thống Multi-Agent theo pattern **Supervisor - Workers** sử dụng LangGraph.

---

## 🏗️ Kiến Trúc

```
User Question
      ↓
  Supervisor (LLM)
  ┌──────────────────────────────┐
  │ Đọc câu hỏi + context       │
  │ Quyết định Worker tiếp theo │
  └──┬────────────┬─────────────┘
     │            │            │
     ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────────────┐
│  Worker 1│ │  Worker 2│ │     Worker 3     │
│  Legal   │ │  News    │ │   PageIndex      │
│          │ │          │ │   (Fallback)     │
│ Pháp luật│ │ Nghệ sĩ  │ │ Vectorless API   │
│ ma tuý   │ │ ma tuý   │ │                  │
└────┬─────┘ └────┬─────┘ └────────┬─────────┘
     │             │               │
     └─────────────┴───────────────┘
                   │ context += results
                   ▼
          Supervisor (vòng tiếp)
                   │ khi đủ info → FINISH
                   ▼
         Synthesizer (LLM)
         → Câu trả lời cuối
           có trích dẫn [Nguồn: X]
```

### Ba Workers

| Worker | Mô tả | Sử dụng |
|--------|--------|---------|
| **legal_worker** | Tìm văn bản pháp luật ma tuý (Điều 249 BLHS, Luật PCMT 2021...) | Task 9 Retrieval Pipeline (hybrid search) |
| **news_worker** | Tìm bài báo về nghệ sĩ liên quan ma tuý | Task 9 Retrieval Pipeline (news-focused) |
| **pageindex_worker** | Fallback khi hybrid search không đủ context | Task 8 PageIndex vectorless API |

---

## 🔧 Công Nghệ

| Layer | Công nghệ |
|-------|-----------|
| Multi-Agent Orchestration | **LangGraph** (StateGraph, conditional_edges) |
| LLM | **Ollama** (llama3.1, local) / OpenAI / Gemini |
| Vector Search | **Weaviate** Cloud (dense + BM25 hybrid) |
| Reranking | **Jina** cross-encoder API |
| Fallback Search | **PageIndex** vectorless API |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Web UI | **Streamlit** |

---

## 📁 Cấu Trúc Thư Mục

```
Lab_Assignment/
├── .env                        ← API keys & cấu hình
├── agent.py                    ← CORE: Supervisor-Workers LangGraph agent
├── app.py                      ← Streamlit chatbot UI
├── test_agent.py               ← Test script (3 câu hỏi mẫu)
├── README.md                   ← File này
│
├── src/                        ← Modules RAG từ Day 08
│   ├── __init__.py
│   ├── config.py               ← Đọc .env, cấu hình global
│   ├── task4_chunking_indexing.py  ← Chunking + Index vào Weaviate
│   ├── task5_semantic_search.py    ← Dense vector search
│   ├── task6_lexical_search.py     ← BM25 keyword search
│   ├── task7_reranking.py          ← Jina/MMR/RRF reranking
│   ├── task8_pageindex_vectorless.py ← PageIndex fallback
│   ├── task9_retrieval_pipeline.py  ← Pipeline tổng hợp (Tasks 5-8)
│   ├── task10_generation.py         ← Generation có citation
│   └── utils_text.py               ← Tokenize, embed, cosine sim
│
└── data/
    ├── landing/legal/          ← PDF/DOCX văn bản pháp luật
    ├── landing/news/           ← HTML/JSON bài báo
    ├── standardized/           ← Markdown đã convert (Task 3)
    └── local_index.json        ← Offline fallback index
```

---

## 🚀 Hướng Dẫn Chạy

### 1. Cấu hình môi trường

Chỉnh file `.env` trong thư mục này. Mặc định dùng **Ollama** (local):

```env
LLM_PROVIDER=ollama        # hoặc 'openai' / 'gemini'
OLLAMA_MODEL=llama3.1:latest
VECTOR_STORE=weaviate       # hoặc 'local' để chạy offline
```

### 2. Chạy Agent qua console

```bash
# Từ thư mục Lab_Assignment
python agent.py "Hình phạt tội tàng trữ ma tuý là gì?"
```

### 3. Kiểm thử 3 loại câu hỏi

```bash
python test_agent.py
```

Test cases:
- 🔵 Câu hỏi pháp luật → `legal_worker`
- 🟣 Câu hỏi nghệ sĩ → `news_worker`
- 🟢 Câu hỏi kết hợp → cả hai workers

### 4. Giao diện Streamlit

```bash
streamlit run app.py
```

---

## 💡 Luồng Chi Tiết

### Bước 1 — Supervisor quyết định
LLM phân tích câu hỏi + context đã có → trả về một trong:
- `legal_worker` → câu hỏi về luật, hình phạt, điều khoản
- `news_worker` → câu hỏi về nghệ sĩ, sự kiện, bài báo
- `pageindex_worker` → khi context chưa đủ sau 2 vòng
- `FINISH` → đã đủ context hoặc đã qua ≥3 vòng

### Bước 2 — Worker thu thập context
Mỗi worker gọi `retrieve()` từ `task9_retrieval_pipeline.py`:
1. `semantic_search()` — Dense retrieval (Weaviate near_vector)
2. `lexical_search()` — BM25 keyword matching
3. `rerank_rrf()` — Merge bằng Reciprocal Rank Fusion
4. `rerank()` — Jina cross-encoder reranking
5. Nếu score < 0.3 → `pageindex_search()` fallback

### Bước 3 — Synthesizer tổng hợp
LLM nhận toàn bộ context → tạo câu trả lời:
- Trích dẫn từng nguồn: `[Nguồn: luat-phong-chong-ma-tuy-2021]`
- Nếu thiếu evidence: `"Tôi không thể xác minh điều này"`

---

## 📊 So Sánh Với Bài Gốc (Day 08 Chatbot)

| | Day 08 Chatbot | Day 09 Assignment |
|--|--|--|
| **Pattern** | Single RAG pipeline | Supervisor - Workers |
| **LLM calls** | 1 (generation only) | 2+ (supervisor + synthesis) |
| **Workers** | 1 retriever | 3 specialized workers |
| **Routing** | Manual/keyword | LLM-driven dynamic routing |
| **Fallback** | PageIndex trong pipeline | Dedicated pageindex_worker |
| **Adaptability** | Fixed flow | Adaptive per-question |
