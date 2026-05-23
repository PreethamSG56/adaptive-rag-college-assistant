# College Notes Adaptive RAG Assistant

> An intelligent AI-powered assistant that helps students interact with their college notes using **Retrieval-Augmented Generation (RAG)** with adaptive inference, **handwritten note OCR + NLP**, and a **hallucination guard** to ensure answer accuracy.

---

## ✨ What's New — NLP + Handwritten Notes Integration

| Feature | Description |
|---|---|
| ✍️ **Handwritten Notes OCR** | Upload photos/scans of handwritten notes (PNG, JPG) — EasyOCR reads them offline |
| 🧹 **NLP Post-Processing** | Noise removal → Spell-correction (SymSpell) → Sentence segmentation (spaCy) |
| 📊 **OCR Confidence Scoring** | Every page gets a quality score; low-confidence sources are flagged in the UI |
| 🔑 **Keyword Enrichment** | spaCy extracts named entities & noun chunks → appended to chunks for better BM25 retrieval |
| 🛡️ **Hallucination Guard** | Every LLM answer is verified sentence-by-sentence against retrieved context (token F1) |
| 🚦 **Grounding Verdicts** | Each answer shows `✅ PASS / ⚠️ PARTIAL / 🚫 FAIL` with confidence % in the UI |

---

## Features

| Feature | Description |
|---|---|
| **Document Upload** | PDF, DOCX, PPTX, TXT, PNG, JPG (handwritten) |
| **Hybrid Retrieval** | FAISS semantic search + BM25 keyword search |
| **Adaptive Top-K** | Dynamic retrieval depth based on query complexity |
| **Re-ranking** | Relevance re-ranking of retrieved chunks |
| **Study Modes** | Q&A, Summarize, Exam Questions, Viva, Formulas, Explain |
| **Feedback Loop** | Latency tracking with automatic adjustment |
| **Query Cache** | Exact-match cache for repeated queries |
| **Groq LLM** | Free Llama 3 via Groq API (pre-configured) |

---

## Quick Start

### 1. Create & activate a virtual environment
```powershell
cd "g:\indicnode assaignment\college-notes-rag"

# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat
```

### 2. Install all dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download the spaCy language model (one-time)
```powershell
python -m spacy download en_core_web_sm
```

> Or run the all-in-one setup script:
> ```powershell
> python setup_nlp.py
> ```

### 4. Configure API key
Copy `.env.example` to `.env` and add your Groq key (free):
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at: https://console.groq.com

### 5. Run the app
```powershell
streamlit run app.py
```

---

## Using Handwritten Notes

1. Take a **photo or scan** of your handwritten notes (PNG / JPG)
2. Upload via the **Upload Study Material** panel
3. Click **Index Documents** — the system will:
   - Run **EasyOCR** to extract text (offline, no API needed)
   - Apply **NLP cleaning**: noise removal → spell-correction → sentence segmentation
   - Extract **keywords** via spaCy for better search
   - Store **OCR confidence** per page
4. Ask questions — answers are automatically verified against your notes

---

## System Architecture

```
User Uploads Notes (PDF / DOCX / PPTX / TXT / PNG / JPG)
      ↓
Document Loader (src/ingestion.py)
  ├─ Text files → direct load + spaCy keyword extraction
  ├─ PDFs → PyMuPDF; scanned pages → EasyOCR
  └─ Images → EasyOCR (handwritten support)
      ↓
NLP Post-Processing (src/nlp_processor.py)
  ├─ Noise removal (garbage chars, broken Unicode)
  ├─ Line de-hyphenation
  ├─ Spell correction (SymSpell — offline dictionary)
  ├─ Sentence segmentation (spaCy)
  └─ Keyword / entity extraction → chunk enrichment
      ↓
Text Chunking (500 chars, 50 overlap)
  + OCR confidence & keywords stored in chunk metadata
      ↓
Embedding Generation (all-MiniLM-L6-v2)
      ↓
FAISS Vector Database  +  BM25 Index (keyword-enriched)
      ↓
Adaptive Decision Layer (src/adaptive_selector.py)
      ↓
Hybrid Retrieval + Re-ranking (src/retriever.py)
      ↓
LLM Response Generation (Groq / OpenAI / Local flan-t5)
      ↓
🛡️ Hallucination Guard (src/hallucination_guard.py)
  ├─ Sentence-level token-F1 grounding check
  ├─ Strips / flags unverified claims
  ├─ OCR quality warning if source confidence is low
  └─ Verdict: PASS / PARTIAL / FAIL
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

## Hallucination Guard

Every generated answer is post-processed through `src/hallucination_guard.py`:

1. **Sentence splitting** — answer is broken into individual claims
2. **Token F1 grounding** — each sentence compared to the retrieved context corpus
3. **Verdict assignment:**
   - `✅ PASS` — ≥50% of sentences grounded in your notes
   - `⚠️ PARTIAL` — 25–50% grounded; unverified claims removed with a caveat
   - `🚫 FAIL` — <25% grounded; replaced with "not found in notes" message
4. **OCR warning** — if source pages had low OCR confidence, a reminder is appended

---

## Folder Structure

```
college-notes-rag/
├── app.py                         # Streamlit UI
├── requirements.txt
├── setup_nlp.py                   # One-shot NLP setup & self-test script
├── .env.example                   # API key template
├── .env                           # API keys (not committed)
├── src/
│   ├── ingestion.py               # Document loading, OCR, NLP enrichment
│   ├── nlp_processor.py           # ✨ NEW: NLP post-processing pipeline
│   ├── hallucination_guard.py     # ✨ NEW: Answer grounding & verification
│   ├── vector_store.py            # FAISS index
│   ├── keyword_store.py           # BM25 index
│   ├── retriever.py               # Hybrid retrieval + re-ranking
│   ├── adaptive_selector.py       # Adaptive decision layer
│   ├── generator.py               # LLM generation (Groq/OpenAI/Local)
│   ├── feedback.py                # Metrics & feedback loop
│   ├── cache.py                   # Query cache
│   └── pipeline.py                # Main orchestrator
├── sample_notes/                  # Demo study material
│   ├── operating_systems.txt
│   ├── dbms.txt
│   └── computer_networks.txt
├── venv/                          # Local virtual environment (not committed)
└── embeddings/                    # Auto-created FAISS index (not committed)
```

---

## Performance Metrics

The system tracks per-query:
- Retrieval time vs Generation time
- P50 / P95 latency
- Strategy distribution (vector/hybrid/keyword)
- Cache hit rate
- Grounding confidence score per answer

View stats in the sidebar after running queries.
