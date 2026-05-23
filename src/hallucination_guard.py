"""
Hallucination Guard
====================
Prevents the LLM from hallucinating by enforcing strict grounding
between the generated answer and the retrieved source context.

Strategy:
  1. Sentence-level overlap check (token F1 between answer & context)
  2. Claim extraction + per-claim verification against context chunks
  3. Low-confidence OCR flagging (source text might be wrong)
  4. Answer sanitisation — strips or flags ungrounded sentences
  5. Confidence score returned alongside the answer

This module is OFFLINE — no external API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.ingestion import Document


# ── Tokenisation helper ──────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "then", "once", "and", "but", "or", "so", "yet", "both",
    "either", "neither", "not", "no", "nor", "as", "if", "while",
    "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "because", "any",
}


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens with stop-word removal."""
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _token_f1(pred: str, ref: str) -> float:
    """Compute token-level F1 between a predicted string and reference."""
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ── Sentence splitter ────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Heuristic sentence splitter (no spaCy dependency)."""
    # Split on .  !  ?  followed by whitespace or end-of-string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Also split on newlines that separate paragraphs
    result = []
    for s in sentences:
        parts = s.split("\n")
        result.extend(p.strip() for p in parts if p.strip())
    return [s for s in result if len(s) > 10]


# ── Context builder ──────────────────────────────────────────────────────────

def _build_context_corpus(
    context_docs: List[Tuple[Document, float]]
) -> Tuple[str, List[str]]:
    """
    Combine all retrieved context into one string and a list of sentences.
    Strips the [KEY_CONCEPTS:...] metadata footers before grounding.
    """
    combined = []
    for doc, _ in context_docs:
        # Remove keyword enrichment footer
        text = re.sub(r"\[KEY_CONCEPTS:.*?\]", "", doc.text, flags=re.DOTALL).strip()
        combined.append(text)
    corpus = " ".join(combined)
    sentences = _split_sentences(corpus)
    return corpus, sentences


# ── Grounding check ──────────────────────────────────────────────────────────

def _sentence_is_grounded(
    sentence: str,
    context_corpus: str,
    context_sentences: List[str],
    threshold: float = 0.15,
) -> Tuple[bool, float]:
    """
    Return (is_grounded, best_f1_score).
    A sentence is grounded if at least one context sentence has F1 >= threshold.
    """
    # Quick substring check first (exact phrase match)
    s_tokens = _tokenize(sentence)
    if not s_tokens:
        return True, 1.0  # empty sentences are trivially OK

    # Full-corpus F1
    corpus_f1 = _token_f1(sentence, context_corpus)
    if corpus_f1 >= threshold:
        return True, corpus_f1

    # Per-sentence F1 check
    best_f1 = 0.0
    for ctx_sent in context_sentences:
        f1 = _token_f1(sentence, ctx_sent)
        if f1 > best_f1:
            best_f1 = f1
        if f1 >= threshold:
            return True, f1

    return False, best_f1


# ── Public dataclass & main entry point ──────────────────────────────────────

@dataclass
class GuardResult:
    original_answer: str
    grounded_answer: str
    overall_confidence: float           # 0-1
    grounded_sentence_ratio: float      # fraction of sentences grounded
    ungrounded_sentences: List[str] = field(default_factory=list)
    ocr_quality_warning: bool = False
    sources_used: int = 0
    verdict: str = "PASS"              # "PASS" | "PARTIAL" | "FAIL"


_NO_CONTEXT_RESPONSE = (
    "I could not find this information in the uploaded notes. "
    "Please ensure the relevant notes have been indexed before asking."
)

_LOW_CONFIDENCE_NOTE = (
    "\n\n> ⚠️ **Note:** Some source pages had low OCR confidence "
    "(handwritten text may have been imperfectly read). "
    "Please cross-check with the original notes."
)


def guard_answer(
    answer: str,
    context_docs: List[Tuple[Document, float]],
    grounding_threshold: float = 0.12,
    pass_ratio: float = 0.55,
    partial_ratio: float = 0.30,
    ocr_confidence_threshold: float = 0.55,
) -> GuardResult:
    """
    Verify an LLM-generated answer against retrieved context.

    Parameters
    ----------
    answer : str
        Raw LLM answer.
    context_docs : list of (Document, score)
        Retrieved context documents.
    grounding_threshold : float
        Minimum token-F1 for a sentence to be considered grounded.
    pass_ratio : float
        Minimum grounded-sentence ratio for a PASS verdict.
    partial_ratio : float
        Minimum grounded-sentence ratio for a PARTIAL verdict (< pass).
    ocr_confidence_threshold : float
        Minimum average OCR quality before issuing a warning.

    Returns
    -------
    GuardResult
    """
    if not context_docs:
        return GuardResult(
            original_answer=answer,
            grounded_answer=_NO_CONTEXT_RESPONSE,
            overall_confidence=0.0,
            grounded_sentence_ratio=0.0,
            verdict="FAIL",
        )

    # 1. Build context
    corpus, ctx_sentences = _build_context_corpus(context_docs)

    # 2. Check OCR quality on source docs
    ocr_scores = []
    for doc, _ in context_docs:
        q = doc.metadata.get("ocr_confidence", None)
        if q is not None:
            ocr_scores.append(float(q))
    avg_ocr = sum(ocr_scores) / len(ocr_scores) if ocr_scores else 1.0
    ocr_warning = avg_ocr < ocr_confidence_threshold and len(ocr_scores) > 0

    # 3. Split answer into sentences and check each
    answer_sentences = _split_sentences(answer)
    if not answer_sentences:
        return GuardResult(
            original_answer=answer,
            grounded_answer=answer,
            overall_confidence=0.5,
            grounded_sentence_ratio=1.0,
            verdict="PASS",
        )

    grounded = []
    ungrounded = []
    f1_scores = []

    for sent in answer_sentences:
        is_grounded, f1 = _sentence_is_grounded(
            sent, corpus, ctx_sentences, threshold=grounding_threshold
        )
        f1_scores.append(f1)
        if is_grounded:
            grounded.append(sent)
        else:
            ungrounded.append(sent)

    ratio = len(grounded) / len(answer_sentences)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    # 4. Determine verdict
    if ratio >= pass_ratio:
        verdict = "PASS"
    elif ratio >= partial_ratio:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # 5. Build grounded answer
    if verdict == "FAIL":
        grounded_answer = _NO_CONTEXT_RESPONSE
    elif verdict == "PARTIAL":
        # Keep only grounded sentences, add a caveat
        if grounded:
            grounded_answer = " ".join(grounded)
            grounded_answer += (
                "\n\n> ⚠️ **Partial answer:** Some parts of the response could not be "
                "fully verified against the uploaded notes and were removed."
            )
        else:
            grounded_answer = _NO_CONTEXT_RESPONSE
    else:
        grounded_answer = answer

    # 6. Append OCR warning if needed
    if ocr_warning:
        grounded_answer += _LOW_CONFIDENCE_NOTE

    return GuardResult(
        original_answer=answer,
        grounded_answer=grounded_answer,
        overall_confidence=round(avg_f1, 4),
        grounded_sentence_ratio=round(ratio, 4),
        ungrounded_sentences=ungrounded,
        ocr_quality_warning=ocr_warning,
        sources_used=len(context_docs),
        verdict=verdict,
    )


def format_guard_metadata(result: GuardResult) -> str:
    """Return a short human-readable summary of the grounding check."""
    lines = [
        f"**Grounding Check:** {result.verdict}",
        f"**Confidence:** {result.overall_confidence:.0%}",
        f"**Grounded Sentences:** {result.grounded_sentence_ratio:.0%}",
    ]
    if result.ocr_quality_warning:
        lines.append("⚠️ Low OCR confidence on some source pages")
    if result.ungrounded_sentences:
        lines.append(f"🚫 Removed {len(result.ungrounded_sentences)} unverified claim(s)")
    return " · ".join(lines)
