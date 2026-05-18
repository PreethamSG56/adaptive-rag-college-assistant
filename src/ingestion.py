"""
Document Loader and Chunker
============================
Supports: PDF, DOCX, PPTX, TXT files.
Chunks text with configurable size and overlap.
"""

from __future__ import annotations

import os
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _transcribe_image_local(image_bytes: bytes) -> str:
    """Uses offline EasyOCR (CNN/RNN based) to read handwritten or scanned text."""
    try:
        import easyocr
        import warnings
        # Suppress FutureWarnings from torch/easyocr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        print("[OCR] Running EasyOCR on image... this might take a moment.")
        results = reader.readtext(image_bytes, detail=0)
        return " ".join(results).strip()
    except Exception as e:
        print(f"[OCR] EasyOCR error: {e}")
        return ""


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


def _load_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF. Scanned pages are automatically OCR'd via EasyOCR."""
    full_text = ""
    # Try PyMuPDF (fitz) first
    try:
        import fitz
        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            
            # If the page has little extractable text, it's likely a scanned handwritten page
            if len(text) < 150:
                print(f"[Loader] Page {i+1} appears to be scanned. Running OCR...")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better OCR
                img_bytes = pix.tobytes("png")
                ocr_text = _transcribe_image_local(img_bytes)
                if ocr_text:
                    text = ocr_text
                    
            if text:
                pages.append(text)
        full_text = "\n\n".join(pages)
        print(f"[Loader] PDF loaded via PyMuPDF: {len(full_text)} chars from {len(pages)} pages")
        return full_text
    except Exception as e:
        print(f"[Loader] PyMuPDF error {path.name}: {e}")

    # Fallback to pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n\n".join(pages)
        print(f"[Loader] PDF loaded via pdfplumber fallback: {len(full_text)} chars")
        return full_text
    except Exception as e:
        print(f"[Loader] pdfplumber error {path.name}: {e}")

    # Final fallback to pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(pages)
        print(f"[Loader] PDF loaded via pypdf fallback: {len(full_text)} chars")
        return full_text
    except Exception as e:
        print(f"[Loader] pypdf error {path.name}: {e}")
        return ""


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


def _load_image(path: Path) -> str:
    """Extract handwritten or printed text from an image using offline EasyOCR."""
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        print(f"[Loader] Running OCR on image {path.name}...")
        return _transcribe_image_local(img_bytes)
    except Exception as e:
        print(f"[Loader] Image OCR error {path.name}: {e}")
        return ""


def load_file(path: Path) -> str:
    """Dispatch to the correct loader based on file extension."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _load_txt(path)
    elif ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".docx":
        return _load_docx(path)
    elif ext in (".pptx", ".ppt"):
        return _load_pptx(path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return _load_image(path)
    else:
        return ""


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
        raw_text = load_file(file_path)
        if not raw_text.strip():
            print(f"[Loader] Empty or unreadable: {file_path.name}")
            continue
        chunks = chunk_text(raw_text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    text=chunk,
                    source=file_path.name,
                    chunk_id=i,
                    metadata={"file": str(file_path)},
                )
            )
        print(f"[Loader] {file_path.name} -> {len(chunks)} chunks")

    print(f"[Loader] Total documents: {len(documents)}")
    return documents
