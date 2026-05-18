"""
BM25 Keyword Store
===================
Sparse keyword retrieval using rank_bm25.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.ingestion import Document


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class KeywordStore:
    def __init__(self):
        self._bm25: BM25Okapi = None
        self._documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        self._documents.extend(documents)
        tokenized = [_tokenize(doc.text) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[KeywordStore] BM25 indexed {len(self._documents)} docs.")

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if self._bm25 is None or not self._documents:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        results = []
        for idx, score in ranked[:top_k]:
            results.append((self._documents[idx], float(score)))
        return results

    def clear(self) -> None:
        self._bm25 = None
        self._documents = []

    @property
    def doc_count(self) -> int:
        return len(self._documents)
