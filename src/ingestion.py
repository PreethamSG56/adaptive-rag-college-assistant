"""
Document Loader and Chunker
============================
Supports: PDF, DOCX, PPTX, TXT, PNG, JPG files.
Chunks text with configurable size and overlap.

NLP Pipeline (for handwritten / scanned notes):
  OCR (EasyOCR)  →  Noise removal  →  Spell-correction (SymSpell)
  →  Sentence segmentation (spaCy)  →  Keyword extraction
  →  Confidence scoring  →  Chunk enrichment
"""

from __future__ import annotations

import os
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _transcribe_image_local(image_bytes: bytes, run_nlp: bool = True) -> tuple:
    """
    Uses offline EasyOCR (CNN/RNN) to read handwritten or scanned text,
    then applies NLP post-processing to clean and enrich the result.

    Returns
    -------
    tuple (cleaned_text: str, ocr_confidence: float, keywords: list[str])
    """
    raw_text = ""
    try:
        import easyocr
        import warnings
        # Suppress FutureWarnings from torch/easyocr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        print("[OCR] Running EasyOCR on image... this might take a moment.")
        results = reader.readtext(image_bytes, detail=0)
        raw_text = " ".join(results).strip()
        print(f"[OCR] Raw text extracted: {len(raw_text)} chars")
    except Exception as e:
        print(f"[OCR] EasyOCR error: {e}")
        return "", 0.0, []

    if not raw_text:
        return "", 0.0, []

    # ── NLP post-processing ──
    if run_nlp:
        try:
            from src.nlp_processor import process_ocr_text
            nlp_result = process_ocr_text(
                raw_text,
                run_spell_correction=True,
                run_spacy=True,
                is_handwritten=True,
            )
            print(
                f"[NLP] OCR confidence: {nlp_result.ocr_confidence:.2f} | "
                f"spell_corrected={nlp_result.spell_corrected} | "
                f"spacy={nlp_result.spacy_enriched} | "
                f"keywords={len(nlp_result.keywords)}"
            )
            return nlp_result.cleaned_text, nlp_result.ocr_confidence, nlp_result.keywords
        except Exception as e:
            print(f"[NLP] Post-processing error: {e} — using raw OCR text")

    return raw_text, 0.5, []


@dataclass
class Document:
    text: str
    source: str
    page: int = 0
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)


def _load_txt(path: Path) -> str:
    """Load plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: Path) -> tuple:
    """
    Extract text from PDF using PyMuPDF.
    Scanned/handwritten pages are automatically OCR'd via EasyOCR + NLP pipeline.

    Returns
    -------
    tuple (text: str, page_meta: list[dict])
        page_meta contains per-page ocr_confidence and keywords.
    """
    page_meta: list[dict] = []

    # Try PyMuPDF (fitz) first
    try:
        import fitz
        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            page_ocr_conf = 1.0
            page_keywords: list = []

            # If the page has little extractable text, it's likely scanned/handwritten
            if len(text) < 150:
                print(f"[Loader] Page {i+1} appears to be scanned/handwritten. Running OCR + NLP...")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_bytes = pix.tobytes("png")
                ocr_text, conf, kws = _transcribe_image_local(img_bytes)
                if ocr_text:
                    text = ocr_text
                    page_ocr_conf = conf
                    page_keywords = kws

            if text:
                pages.append(text)
                page_meta.append({
                    "page": i + 1,
                    "ocr_confidence": page_ocr_conf,
                    "keywords": page_keywords,
                })

        full_text = "\n\n".join(pages)
        print(f"[Loader] PDF loaded via PyMuPDF: {len(full_text)} chars from {len(pages)} pages")
        return full_text, page_meta
    except Exception as e:
        print(f"[Loader] PyMuPDF error {path.name}: {e}")

    # Fallback to pdfplumber (no OCR meta)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n\n".join(pages)
        print(f"[Loader] PDF loaded via pdfplumber fallback: {len(full_text)} chars")
        return full_text, []
    except Exception as e:
        print(f"[Loader] pdfplumber error {path.name}: {e}")

    # Final fallback to pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(pages)
        print(f"[Loader] PDF loaded via pypdf fallback: {len(full_text)} chars")
        return full_text, []
    except Exception as e:
        print(f"[Loader] pypdf error {path.name}: {e}")
        return "", []


def _load_docx(path: Path) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document as DocxDoc
        doc = DocxDoc(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[Loader] DOCX error {path.name}: {e}")
        return ""


def _load_pptx(path: Path) -> str:
    """Extract text from PPTX file."""
    try:
        from pptx import Presentation
        prs = Presentation(str(path))
        slides = []
        for i, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                slides.append(f"[Slide {i+1}]\n" + "\n".join(texts))
        return "\n\n".join(slides)
    except Exception as e:
        print(f"[Loader] PPTX error {path.name}: {e}")
        return ""


def _load_image(path: Path) -> tuple:
    """
    Extract handwritten or printed text from an image using offline EasyOCR + NLP.

    Returns
    -------
    tuple (text: str, ocr_confidence: float, keywords: list[str])
    """
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        print(f"[Loader] Running OCR + NLP on image {path.name}...")
        return _transcribe_image_local(img_bytes)
    except Exception as e:
        print(f"[Loader] Image OCR error {path.name}: {e}")
        return "", 0.0, []


def load_file(path: Path) -> tuple:
    """
    Dispatch to the correct loader based on file extension.

    Returns
    -------
    tuple (text: str, ocr_confidence: float, keywords: list[str], page_meta: list[dict])
    """
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        # Plain text — apply lightweight NLP for keyword extraction
        raw = _load_txt(path)
        keywords: list = []
        try:
            from src.nlp_processor import _extract_entities_and_keywords
            keywords = _extract_entities_and_keywords(raw)
        except Exception:
            pass
        return raw, 1.0, keywords, []
    elif ext == ".pdf":
        text, page_meta = _load_pdf(path)
        # Aggregate keywords from all pages
        all_kws: list = []
        for pm in page_meta:
            all_kws.extend(pm.get("keywords", []))
        avg_conf = (
            sum(pm.get("ocr_confidence", 1.0) for pm in page_meta) / len(page_meta)
            if page_meta else 1.0
        )
        return text, avg_conf, list(dict.fromkeys(all_kws)), page_meta
    elif ext == ".docx":
        raw = _load_docx(path)
        return raw, 1.0, [], []
    elif ext in (".pptx", ".ppt"):
        raw = _load_pptx(path)
        return raw, 1.0, [], []
    elif ext in (".png", ".jpg", ".jpeg"):
        text, conf, kws = _load_image(path)
        return text, conf, kws, []
    else:
        return "", 1.0, [], []


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping character-level chunks.

    The function first collapses redundant whitespace so that chunk sizes
    are predictable, then slices the text with the requested ``overlap``
    to preserve context at boundaries.

    Parameters
    ----------
    text : str
        Cleaned input text.
    chunk_size : int
        Maximum number of characters per chunk (default 500).
    overlap : int
        Number of characters to repeat at the start of each successive
        chunk so that retrieval is not confused by arbitrary boundaries
        (default 50).

    Returns
    -------
    List[str]
        List of non-empty text chunks.
    """
    # Collapse all whitespace runs to a single space
    text = re.sub(r"[\t ]+", " ", text)          # horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # max two consecutive newlines
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def ingest_documents(
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Document]:
    """
    Load all supported files from a file path or directory.
    Returns a flat list of Document objects.
    """
    source_path = Path(source)
    documents: List[Document] = []

    if source_path.is_dir():
        files = (
            list(source_path.rglob("*.txt"))
            + list(source_path.rglob("*.md"))
            + list(source_path.rglob("*.pdf"))
            + list(source_path.rglob("*.docx"))
            + list(source_path.rglob("*.pptx"))
            + list(source_path.rglob("*.png"))
            + list(source_path.rglob("*.jpg"))
            + list(source_path.rglob("*.jpeg"))
        )
    elif source_path.is_file():
        files = [source_path]
    else:
        print(f"[Loader] Path not found: {source}")
        return []

    for file_path in files:
        print(f"[Loader] Loading: {file_path.name}")
        text, ocr_confidence, keywords, page_meta = load_file(file_path)
        if not text.strip():
            print(f"[Loader] Empty or unreadable: {file_path.name}")
            continue

        # Keyword enrichment for BM25 searchability
        try:
            from src.nlp_processor import enrich_chunk_with_keywords
            enrich = True
        except Exception:
            enrich = False

        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            enriched_chunk = enrich_chunk_with_keywords(chunk, keywords) if enrich else chunk
            documents.append(
                Document(
                    text=enriched_chunk,
                    source=file_path.name,
                    chunk_id=i,
                    metadata={
                        "file": str(file_path),
                        "ocr_confidence": ocr_confidence,
                        "keywords": ",".join(keywords[:10]),
                        "is_handwritten": ocr_confidence < 0.95,
                    },
                )
            )
        print(f"[Loader] {file_path.name} -> {len(chunks)} chunks (OCR conf: {ocr_confidence:.2f})")

    print(f"[Loader] Total documents: {len(documents)}")
    return documents
