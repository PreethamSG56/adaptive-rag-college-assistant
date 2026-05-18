"""
Hybrid Retriever + Re-ranking
================================
Combines FAISS vector search and BM25 keyword search via
weighted score fusion, then optionally re-ranks results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.ingestion import Document
from src.vector_store import VectorStore
from src.keyword_store import KeywordStore


@dataclass
class RetrievalConfig:
    top_k: int = 5
    strategy: str = "hybrid"   # "vector" | "keyword" | "hybrid"
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    rerank: bool = True


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[Tuple[Document, float]]:
        k = max(config.top_k, 3)

        if config.strategy == "vector":
            results = self.vector_store.search(query, top_k=k)
        elif config.strategy == "keyword":
            results = self.keyword_store.search(query, top_k=k)
        else:
            results = self._hybrid_fusion(query, k, config)

        if config.rerank:
            results = self._rerank(query, results)

        return results[: config.top_k]

    # ---- private helpers ------------------------------------------------

    def _hybrid_fusion(
        self,
        query: str,
        k: int,
        config: RetrievalConfig,
    ) -> List[Tuple[Document, float]]:
        vec_results = self.vector_store.search(query, top_k=k * 2)
        kw_results = self.keyword_store.search(query, top_k=k * 2)

        # Normalize BM25 scores to [0, 1]
        bm25_max = max((s for _, s in kw_results), default=1.0) or 1.0

        scores: dict[str, Tuple[Document, float]] = {}

        for doc, score in vec_results:
            key = f"{doc.source}_{doc.chunk_id}"
            scores[key] = (doc, score * config.vector_weight)

        for doc, score in kw_results:
            key = f"{doc.source}_{doc.chunk_id}"
            norm_score = (score / bm25_max) * config.keyword_weight
            if key in scores:
                existing_doc, existing_score = scores[key]
                scores[key] = (existing_doc, existing_score + norm_score)
            else:
                scores[key] = (doc, norm_score)

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return ranked

    def _rerank(
        self,
        query: str,
        results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Simple keyword-overlap re-ranking (no heavy cross-encoder model
        needed for this project size).
        """
        q_words = set(query.lower().split())
        reranked = []
        for doc, score in results:
            doc_words = set(doc.text.lower().split())
            overlap = len(q_words & doc_words) / (len(q_words) + 1e-6)
            reranked.append((doc, score + 0.1 * overlap))
        return sorted(reranked, key=lambda x: x[1], reverse=True)
