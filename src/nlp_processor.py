"""
NLP Post-Processing Pipeline
==============================
Cleans and enriches raw OCR text extracted from handwritten notes.

Steps:
  1. Character-level noise removal (stray symbols, broken Unicode)
  2. Line de-hyphenation and word boundary correction
  3. Spell-correction via SymSpell (fast, dictionary-based, offline)
  4. Sentence segmentation & reconstruction with spaCy (optional)
  5. Keyword / concept extraction for BM25 enrichment tags
  6. OCR confidence scoring (heuristic) — used by HallucinationGuard

All heavy imports are lazy so the module loads instantly even if
optional libraries are not installed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Heuristic OCR quality scorer ────────────────────────────────────────────

def _ocr_quality_score(text: str) -> float:
    """
    Return a 0-1 confidence score for OCR-extracted text.
    High score = likely clean; low score = likely garbled.

    Heuristics:
    - Ratio of alphabetic/digit chars vs total
    - Average word length (too short or too long → garbled)
    - Density of special/garbage characters
    """
    if not text or len(text) < 5:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    # Alpha-numeric ratio
    alnum_count = sum(c.isalnum() or c in " \n.,;:?!" for c in text)
    alnum_ratio = alnum_count / len(text)

    # Average word length (ideal 3-12 chars)
    avg_len = sum(len(w) for w in words) / len(words)
    len_score = 1.0 if 3 <= avg_len <= 12 else max(0.0, 1.0 - abs(avg_len - 7) / 10)

    # Garbage character density (control chars, replacement char, etc.)
    garbage = sum(1 for c in text if unicodedata.category(c) in ("Cc", "Cf", "Cs") or c == "\ufffd")
    garbage_ratio = garbage / max(len(text), 1)

    score = (alnum_ratio * 0.5) + (len_score * 0.4) + ((1.0 - min(garbage_ratio * 10, 1.0)) * 0.1)
    return round(min(max(score, 0.0), 1.0), 3)


# ── Character-level noise removal ───────────────────────────────────────────

_GARBAGE_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]"  # control chars
    r"|[^\x00-\x7f\u00a0-\u024f\u0900-\u097f]",  # non-Latin/Devanagari beyond basic multilingual
)

def _remove_noise(text: str) -> str:
    """Strip garbage characters that OCR frequently produces."""
    text = _GARBAGE_PATTERN.sub(" ", text)
    # Fix common OCR symbol substitutions
    text = text.replace("|", "I")          # pipe → capital I
    text = text.replace("0", "O") if _looks_like_word_context(text) else text
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_word_context(text: str) -> bool:
    """Rough check — True if the text contains mostly letters."""
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) > 0.6


# ── Hyphenation / line break repair ─────────────────────────────────────────

def _dehyphenate(text: str) -> str:
    """
    Join words broken across lines with a hyphen:
    'algo-\\nrithm' → 'algorithm'
    """
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)
    return text


# ── Spell correction (SymSpell — optional) ───────────────────────────────────

_symspell_instance = None

def _get_symspell():
    """Lazy-load SymSpell with a bundled frequency dictionary."""
    global _symspell_instance
    if _symspell_instance is not None:
        return _symspell_instance
    try:
        from symspellpy import SymSpell, Verbosity  # noqa: F401
        import pkg_resources

        ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )
        ss.load_dictionary(dict_path, term_index=0, count_index=1)
        ss.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
        _symspell_instance = ss
        print("[NLP] SymSpell dictionary loaded.")
    except Exception as e:
        print(f"[NLP] SymSpell not available: {e} — skipping spell correction.")
        _symspell_instance = None
    return _symspell_instance


def _spell_correct(text: str, max_words: int = 1000) -> str:
    """
    Apply SymSpell compound correction to each line (not individual tokens,
    to preserve context).  Silently skips if SymSpell is unavailable.
    """
    ss = _get_symspell()
    if ss is None:
        return text

    try:
        from symspellpy import Verbosity
        lines = text.split("\n")
        corrected_lines = []
        for line in lines:
            words = line.split()
            if len(words) > max_words or not line.strip():
                corrected_lines.append(line)
                continue
            suggestions = ss.lookup_compound(line, max_edit_distance=2)
            if suggestions:
                corrected_lines.append(suggestions[0].term)
            else:
                corrected_lines.append(line)
        return "\n".join(corrected_lines)
    except Exception as e:
        print(f"[NLP] Spell correction error: {e}")
        return text


# ── spaCy NLP enrichment (optional) ─────────────────────────────────────────

_nlp_model = None

def _get_spacy():
    """Lazy-load the small spaCy model."""
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    try:
        import spacy
        _nlp_model = spacy.load("en_core_web_sm")
        print("[NLP] spaCy en_core_web_sm loaded.")
    except Exception as e:
        print(f"[NLP] spaCy not available: {e} — skipping NLP enrichment.")
        _nlp_model = None
    return _nlp_model


def _extract_entities_and_keywords(text: str) -> List[str]:
    """
    Use spaCy to extract named entities and noun chunks.
    Returns a list of keyword strings for metadata enrichment.
    """
    nlp = _get_spacy()
    if nlp is None:
        # Fallback: simple regex-based noun extraction
        tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        return list(dict.fromkeys(tokens))[:20]

    try:
        doc = nlp(text[:5000])  # limit to avoid OOM on huge pages
        keywords = []
        for ent in doc.ents:
            keywords.append(ent.text.strip())
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text.strip())
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and len(kw) > 2:
                seen.add(kw_lower)
                unique.append(kw)
        return unique[:30]
    except Exception as e:
        print(f"[NLP] Entity extraction error: {e}")
        return []


def _segment_sentences(text: str) -> str:
    """
    Re-segment sentence boundaries using spaCy.
    Helps when OCR misses periods or merges lines incorrectly.
    """
    nlp = _get_spacy()
    if nlp is None:
        return text
    try:
        doc = nlp(text[:10000])
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return " ".join(sentences)
    except Exception as e:
        print(f"[NLP] Sentence segmentation error: {e}")
        return text


# ── Public dataclass & main entry point ─────────────────────────────────────

@dataclass
class NLPResult:
    original_text: str
    cleaned_text: str
    ocr_confidence: float           # 0-1 heuristic score
    keywords: List[str] = field(default_factory=list)
    is_handwritten: bool = False
    spell_corrected: bool = False
    spacy_enriched: bool = False


def process_ocr_text(
    raw_text: str,
    run_spell_correction: bool = True,
    run_spacy: bool = True,
    is_handwritten: bool = True,
) -> NLPResult:
    """
    Full NLP post-processing pipeline for OCR-extracted text.

    Parameters
    ----------
    raw_text : str
        Raw text from EasyOCR or any other OCR engine.
    run_spell_correction : bool
        Whether to apply SymSpell correction (requires symspellpy).
    run_spacy : bool
        Whether to apply spaCy sentence segmentation and entity extraction.
    is_handwritten : bool
        Flag carried through to the result for downstream grounding decisions.

    Returns
    -------
    NLPResult
        Cleaned text + metadata used by the hallucination guard.
    """
    if not raw_text or not raw_text.strip():
        return NLPResult(
            original_text=raw_text or "",
            cleaned_text="",
            ocr_confidence=0.0,
        )

    # Step 1 — basic noise removal
    text = _remove_noise(raw_text)

    # Step 2 — line de-hyphenation
    text = _dehyphenate(text)

    # Step 3 — OCR confidence scoring (on cleaned text)
    confidence = _ocr_quality_score(text)

    # Step 4 — spell correction (optional, offline)
    spell_corrected = False
    if run_spell_correction and confidence > 0.3:  # skip if text is too garbled
        corrected = _spell_correct(text)
        if corrected and corrected != text:
            text = corrected
            spell_corrected = True

    # Step 5 — sentence re-segmentation with spaCy (optional)
    spacy_enriched = False
    if run_spacy:
        segmented = _segment_sentences(text)
        if segmented:
            text = segmented
            spacy_enriched = True

    # Step 6 — keyword / entity extraction
    keywords = _extract_entities_and_keywords(text)

    return NLPResult(
        original_text=raw_text,
        cleaned_text=text,
        ocr_confidence=confidence,
        keywords=keywords,
        is_handwritten=is_handwritten,
        spell_corrected=spell_corrected,
        spacy_enriched=spacy_enriched,
    )


def enrich_chunk_with_keywords(chunk_text: str, keywords: List[str]) -> str:
    """
    Append extracted NLP keywords as a hidden metadata footer to a chunk.
    These are searchable by BM25 but stripped before display.
    """
    if not keywords:
        return chunk_text
    kw_str = ", ".join(keywords[:15])
    return f"{chunk_text}\n[KEY_CONCEPTS: {kw_str}]"
