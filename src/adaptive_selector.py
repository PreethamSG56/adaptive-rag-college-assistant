"""
Adaptive Decision Layer
========================
Analyses query complexity at runtime to dynamically select
optimal retrieval configuration (top-K, strategy, model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.retriever import RetrievalConfig


COMPLEX_KEYWORDS = {
    "explain", "describe", "compare", "difference", "define",
    "summarize", "how", "why", "what", "analyze", "evaluate",
    "discuss", "elaborate", "list", "enumerate", "derive",
    "prove", "demonstrate", "calculate", "solve", "implement",
}

STUDY_KEYWORDS = {
    "exam", "question", "viva", "formula", "summary", "topic",
    "chapter", "unit", "important", "key", "note", "concept",
}


@dataclass
class QueryProfile:
    query: str
    word_count: int
    complexity_score: float
    has_study_intent: bool
    complexity_label: str   # "simple" | "moderate" | "complex"


class AdaptiveSelector:
    """
    Rule-based adaptive selector:
    - Short / simple queries -> top_k=3, vector-only
    - Moderate queries       -> top_k=5, hybrid
    - Complex queries        -> top_k=8, hybrid + rerank
    Latency feedback can further reduce top_k dynamically.
    """

    def __init__(
        self,
        latency_threshold: float = 5.0,
        min_top_k: int = 2,
        max_top_k: int = 10,
    ):
        self.latency_threshold = latency_threshold
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self._recent_latencies: list[float] = []

    # ---- public API -----------------------------------------------------

    def profile_query(self, query: str) -> QueryProfile:
        words = query.strip().split()
        word_count = len(words)
        lower = query.lower()
        complex_hits = sum(1 for w in COMPLEX_KEYWORDS if w in lower)
        study_hits = sum(1 for w in STUDY_KEYWORDS if w in lower)
        # Score: 0-1
        score = min(1.0, (word_count / 20) * 0.5 + (complex_hits / 5) * 0.5)

        if score < 0.3:
            label = "simple"
        elif score < 0.65:
            label = "moderate"
        else:
            label = "complex"

        return QueryProfile(
            query=query,
            word_count=word_count,
            complexity_score=score,
            has_study_intent=study_hits > 0,
            complexity_label=label,
        )

    def select_config(
        self,
        profile: QueryProfile,
        recent_latency: float = 0.0,
    ) -> RetrievalConfig:
        # Base top_k by complexity
        if profile.complexity_label == "simple":
            top_k = 3
            strategy = "vector"
            rerank = False
        elif profile.complexity_label == "moderate":
            top_k = 5
            strategy = "hybrid"
            rerank = True
        else:
            top_k = 8
            strategy = "hybrid"
            rerank = True

        # Latency-aware adjustment
        if recent_latency > self.latency_threshold:
            top_k = max(self.min_top_k, top_k - 2)
            strategy = "vector"
            rerank = False

        # Study intent bumps top_k slightly
        if profile.has_study_intent:
            top_k = min(self.max_top_k, top_k + 1)

        return RetrievalConfig(
            top_k=top_k,
            strategy=strategy,
            rerank=rerank,
        )

    def record_latency(self, latency: float) -> None:
        self._recent_latencies.append(latency)
        if len(self._recent_latencies) > 10:
            self._recent_latencies.pop(0)

    def avg_latency(self) -> float:
        if not self._recent_latencies:
            return 0.0
        return sum(self._recent_latencies) / len(self._recent_latencies)
