"""
College Notes Adaptive RAG Assistant - Streamlit UI
=====================================================
Main application interface for uploading notes and querying them.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Notes RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: rgba(200,200,230,0.8);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(30,27,75,0.8), rgba(49,46,129,0.6));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.metric-card:hover {
    border-color: rgba(168,85,247,0.6);
    transform: translateY(-2px);
}
.metric-number {
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
    margin: 0;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(180,180,210,0.7);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(30,27,75,0.8));
    border: 1px solid rgba(99,102,241,0.45);
    border-left: 4px solid #818cf8;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(8px);
    line-height: 1.7;
    color: rgba(230,230,255,0.95);
}

/* ── Source chip ── */
.source-chip {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.75rem;
    color: #a5b4fc;
    margin: 0.2rem;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-green  { background: rgba(34,197,94,0.2);  color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
.badge-yellow { background: rgba(234,179,8,0.2);  color: #facc15; border: 1px solid rgba(234,179,8,0.4); }
.badge-red    { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.badge-blue   { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid rgba(59,130,246,0.4); }
.badge-purple { background: rgba(168,85,247,0.2); color: #c084fc; border: 1px solid rgba(168,85,247,0.4); }

/* ── Timing row ── */
.timing-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.75rem;
}
.timing-item {
    background: rgba(30,27,75,0.7);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
    color: #a5b4fc;
}

/* ── Section title ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Upload area ── */
.upload-info {
    background: rgba(30,27,75,0.6);
    border: 1px dashed rgba(99,102,241,0.4);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: rgba(180,180,220,0.7);
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,12,41,0.97) 0%, rgba(30,27,75,0.97) 100%);
    border-right: 1px solid rgba(99,102,241,0.25);
}
[data-testid="stSidebar"] * { color: rgba(220,220,250,0.9) !important; }

/* Streamlit widget overrides */
.stTextArea textarea, .stTextInput input {
    background: rgba(30,27,75,0.8) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #9333ea) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}
.stSelectbox > div > div, .stSlider > div {
    background: rgba(30,27,75,0.8) !important;
}
div[data-testid="stFileUploader"] {
    background: rgba(30,27,75,0.6) !important;
    border: 1px dashed rgba(99,102,241,0.5) !important;
    border-radius: 12px !important;
}
.stSpinner > div { border-top-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
def _init_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "indexed_count" not in st.session_state:
        st.session_state.indexed_count = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = None

_init_state()


def get_pipeline():
    """Lazily initialise the pipeline."""
    if st.session_state.pipeline is None:
        from src.pipeline import AdaptiveRAGPipeline
        st.session_state.pipeline = AdaptiveRAGPipeline(
            embeddings_dir="embeddings",
            feedback_log="feedback_log.jsonl",
            use_cache=True,
        )
    return st.session_state.pipeline


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 College Notes RAG")
    st.markdown("---")

    # API Key status
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if groq_key:
        st.markdown('<span class="badge badge-green">✓ Groq API Ready</span>', unsafe_allow_html=True)
    elif openai_key:
        st.markdown('<span class="badge badge-blue">✓ OpenAI API Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">⚠ No API Key — Local Mode</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    study_mode = st.selectbox(
        "Study Mode",
        options=["qa", "summarize", "exam_questions", "viva", "formulas", "explain"],
        format_func=lambda x: {
            "qa": "💬 Q&A",
            "summarize": "📋 Summarize",
            "exam_questions": "📝 Exam Questions",
            "viva": "🎤 Viva Questions",
            "formulas": "🔢 Extract Formulas",
            "explain": "💡 Explain Topic",
        }[x],
        key="study_mode",
    )

    chunk_size = st.slider("Chunk Size (tokens)", 200, 800, 500, 50)
    overlap = st.slider("Chunk Overlap", 0, 150, 50, 10)

    st.markdown("---")
    if st.button("🗑️ Clear All Data"):
        if st.session_state.pipeline:
            st.session_state.pipeline.clear_all()
        st.session_state.pipeline = None
        st.session_state.indexed_count = 0
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.rerun()

    # Stats
    if st.session_state.pipeline:
        stats = st.session_state.pipeline.stats()
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        fb = stats.get("feedback", {})
        if fb:
            st.markdown(f"**Queries:** {fb.get('total_queries', 0)}")
            st.markdown(f"**Avg Time:** {fb.get('avg_total_time', 0)}s")
            st.markdown(f"**P95 Latency:** {fb.get('p95', 0)}s")
        cache_s = stats.get("cache", {})
        if cache_s:
            st.markdown(f"**Cache Hit Rate:** {int(cache_s.get('hit_rate', 0)*100)}%")


# ── Main area ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">🎓 College Notes RAG Assistant</p>
  <p class="hero-subtitle">Upload your study material · Ask questions · Generate exam prep instantly</p>
</div>
""", unsafe_allow_html=True)

# Top metric cards
pipeline = get_pipeline()
stats = pipeline.stats()
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{stats['indexed_docs']}</p>
        <p class="metric-label">Chunks Indexed</p>
    </div>""", unsafe_allow_html=True)
with m2:
    fb = stats.get("feedback", {})
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{fb.get('total_queries', 0)}</p>
        <p class="metric-label">Queries Run</p>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{len(st.session_state.uploaded_files)}</p>
        <p class="metric-label">Files Uploaded</p>
    </div>""", unsafe_allow_html=True)
with m4:
    p50 = fb.get("p50", 0)
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{p50}s</p>
        <p class="metric-label">P50 Latency</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Two-column layout ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

# ── LEFT: Upload ────────────────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-title">📁 Upload Study Material</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-info">
        Supported: <b>PDF, DOCX, PPTX, TXT, PNG, JPG</b><br>
        Drag & drop or click to browse
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload your notes",
        type=["pdf", "docx", "pptx", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⚡ Index Documents", key="index_btn"):
        if not uploaded:
            st.warning("Please upload at least one file first.")
        else:
            # Save to temp dir
            upload_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
            st.session_state.upload_dir = str(upload_dir)

            for uf in uploaded:
                dest = upload_dir / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.read())

            with st.spinner("Parsing and indexing documents..."):
                from src.ingestion import ingest_documents
                from src.pipeline import AdaptiveRAGPipeline

                # Fresh pipeline for new uploads
                pl = AdaptiveRAGPipeline(
                    embeddings_dir="embeddings",
                    feedback_log="feedback_log.jsonl",
                    use_cache=True,
                )
                n = pl.ingest(
                    str(upload_dir),
                    chunk_size=chunk_size,
                    overlap=overlap,
                    force_reload=True,
                )
                st.session_state.pipeline = pl
                st.session_state.indexed_count = n
                st.session_state.uploaded_files = [f.name for f in uploaded]

            st.success(f"✅ Indexed **{n}** chunks from **{len(uploaded)}** file(s)!")
            st.rerun()

    # Uploaded files list
    if st.session_state.uploaded_files:
        st.markdown('<p class="section-title">📄 Indexed Files</p>', unsafe_allow_html=True)
        for fname in st.session_state.uploaded_files:
            ext = fname.split(".")[-1].upper()
            icon = {"PDF": "📕", "DOCX": "📘", "PPTX": "📙", "TXT": "📄", "MD": "📝", "PNG": "🖼️", "JPG": "🖼️", "JPEG": "🖼️"}.get(ext, "📄")
            st.markdown(f"- {icon} `{fname}`")

    # Quick examples
    st.markdown('<p class="section-title">💡 Example Queries</p>', unsafe_allow_html=True)
    examples = [
        "Explain TCP/IP protocol",
        "Summarize DBMS normalization",
        "What is the attention mechanism?",
        "List important formulas",
        "Generate exam questions from Unit 3",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["query_input"] = ex


# ── RIGHT: Chat Interface ───────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="section-title">💬 Ask Your Notes</p>', unsafe_allow_html=True)

    # Query input
    query_val = st.session_state.get("query_input", "")
    query = st.text_area(
        "Your Question",
        value=query_val,
        placeholder="Ask anything about your uploaded study material...\nE.g. 'Explain normalization in DBMS' or 'Generate 10 viva questions'",
        height=110,
        key="main_query",
        label_visibility="collapsed",
    )

    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        ask_btn = st.button("🔍 Ask Assistant", key="ask_btn")
    with clear_col:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Run query
    if ask_btn and query.strip():
        if stats["indexed_docs"] == 0:
            st.warning("⚠️ No documents indexed yet. Please upload and index your study material first.")
        else:
            with st.spinner("🤔 Thinking..."):
                result = pipeline.query(
                    query.strip(),
                    mode=st.session_state.study_mode,
                )
            st.session_state.chat_history.insert(0, result)
            st.session_state["query_input"] = ""
            st.rerun()

    # Chat history
    if st.session_state.chat_history:
        for i, res in enumerate(st.session_state.chat_history):
            # Mode badge
            mode_colors = {
                "qa": "badge-blue", "summarize": "badge-purple",
                "exam_questions": "badge-green", "viva": "badge-yellow",
                "formulas": "badge-red", "explain": "badge-blue",
            }
            mode_labels = {
                "qa": "Q&A", "summarize": "Summary", "exam_questions": "Exam Qs",
                "viva": "Viva Qs", "formulas": "Formulas", "explain": "Explain",
            }
            badge_class = mode_colors.get(res["mode"], "badge-blue")
            mode_label = mode_labels.get(res["mode"], res["mode"])

            from_cache = res.get("from_cache", False)
            cache_tag = ' <span class="badge badge-green">⚡ Cached</span>' if from_cache else ""

            complexity = res["query_profile"]["complexity"]
            complexity_badge = {
                "simple": "badge-green", "moderate": "badge-yellow", "complex": "badge-red"
            }.get(complexity, "badge-blue")

            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <span class="badge {badge_class}">{mode_label}</span>
                <span class="badge {complexity_badge}">{complexity}</span>
                {cache_tag}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**🙋 Query:** {res['query']}")

            # Answer
            st.markdown(f"""
            <div class="answer-box">
                {res['answer'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Sources
            if res["sources"]:
                source_html = "".join(
                    f'<span class="source-chip">📄 {s["source"]} (chunk {s["chunk_id"]}, score: {s["score"]})</span>'
                    for s in res["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

            # Timings
            t = res["timings"]
            cfg = res["retrieval_config"]
            st.markdown(f"""
            <div class="timing-row">
                <span class="timing-item">⏱ Total: {t['total_s']}s</span>
                <span class="timing-item">🔍 Retrieval: {t['retrieval_s']}s</span>
                <span class="timing-item">🤖 Generation: {t['generation_s']}s</span>
                <span class="timing-item">📊 Top-K: {cfg['top_k']}</span>
                <span class="timing-item">🔀 Strategy: {cfg['strategy']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: rgba(160,160,200,0.6);">
            <p style="font-size:3rem; margin:0;">🎓</p>
            <p style="font-size:1.1rem; margin-top:0.5rem;">
                Upload your study notes and ask anything!
            </p>
            <p style="font-size:0.85rem;">
                Supports: Q&A · Summarization · Exam Questions · Viva Prep · Formula Extraction
            </p>
        </div>
        """, unsafe_allow_html=True)
