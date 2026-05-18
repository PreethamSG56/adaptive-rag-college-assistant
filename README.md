# College Notes Adaptive RAG Assistant

> An intelligent AI-powered assistant that helps students interact with their college notes using **Retrieval-Augmented Generation (RAG)** with adaptive inference optimization.

---

## Features

| Feature | Description |
|---|---|
| **Document Upload** | PDF, DOCX, PPTX, TXT support |
| **Hybrid Retrieval** | FAISS semantic search + BM25 keyword search |
| **Adaptive Top-K** | Dynamic retrieval depth based on query complexity |
| **Re-ranking** | Relevance re-ranking of retrieved chunks |
| **Study Modes** | Q&A, Summarize, Exam Questions, Viva, Formulas, Explain |
| **Feedback Loop** | Latency tracking with automatic adjustment |
| **Query Cache** | Exact-match cache for repeated queries |
| **Groq LLM** | Free Llama 3 via Groq API (pre-configured) |

---

## Quick Start

### 1. Install dependencies
```powershell
cd "g:\indicnode assaignment\college-notes-rag"
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```powershell
streamlit run app.py
```

### 3. Use the app
1. Upload your PDF/DOCX/PPTX/TXT notes in the left panel
2. Click **Index Documents**
3. Type your question or pick a study mode
4. Click **Ask Assistant**

---

## System Architecture

```
User Uploads Notes
      ↓
Document Loader (src/ingestion.py)
      ↓
Text Chunking (500 tokens, 50 overlap)
      ↓
Embedding Generation (all-MiniLM-L6-v2)
      ↓
FAISS Vector Database  +  BM25 Index
      ↓
Adaptive Decision Layer (src/adaptive_selector.py)
      ↓
Hybrid Retrieval + Re-ranking (src/retriever.py)
      ↓
LLM Response Generation (Groq / OpenAI / Local)
      ↓
Feedback & Metrics Tracking (src/feedback.py)
```

---

## Study Modes

| Mode | Prompt Style |
|---|---|
| Q&A | Direct question answering from notes |
| Summarize | Structured summary for revision |
| Exam Questions | 10 exam questions from content |
| Viva Questions | 10 deep viva questions |
| Formulas | Extract all equations & definitions |
| Explain | Beginner-friendly explanation |

---

## Adaptive Logic

**Query Complexity Detection:**
```python
if word_count < 5 and no complex keywords:
    top_k = 3, strategy = "vector"      # simple
elif word_count < 15:
    top_k = 5, strategy = "hybrid"     # moderate
else:
    top_k = 8, strategy = "hybrid"     # complex
```

**Latency-Aware Adjustment:**
```python
if avg_recent_latency > 8.0 seconds:
    top_k = max(2, top_k - 2)
    strategy = "vector"  # faster
```

---

## API Key

The `.env` file is pre-configured with a Groq API key (Llama 3, free tier).

To use your own key, edit `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at: https://console.groq.com

---

## Performance Metrics

The system tracks per-query:
- Retrieval time vs Generation time
- P50 / P95 latency
- Strategy distribution (vector/hybrid/keyword)
- Cache hit rate

View stats in the sidebar after running queries.

---

## Folder Structure

```
college-notes-rag/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env                      # API keys (pre-filled)
├── src/
│   ├── ingestion.py          # Document loading & chunking
│   ├── vector_store.py       # FAISS index
│   ├── keyword_store.py      # BM25 index
│   ├── retriever.py          # Hybrid retrieval + re-ranking
│   ├── adaptive_selector.py  # Adaptive decision layer
│   ├── generator.py          # LLM generation (Groq/OpenAI/Local)
│   ├── feedback.py           # Metrics & feedback loop
│   ├── cache.py              # Query cache
│   └── pipeline.py           # Main orchestrator
├── sample_notes/             # Demo study material
│   ├── operating_systems.txt
│   ├── dbms.txt
│   └── computer_networks.txt
└── embeddings/               # Auto-created FAISS index
```
