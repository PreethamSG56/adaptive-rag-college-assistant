"""
FAISS Vector Store
===================
Stores sentence-transformer embeddings and supports
semantic similarity search.

Speed optimisations:
  - Embedding model pre-warmed in a background thread at init time
  - IVFFlat index for large collections (faster search)
  - Batch size 64 for fast encoding
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from src.ingestion import Document


class VectorStore:
    # BAAI/bge-small-en-v1.5 — 384-dim, fast, high quality
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, index_dir: str = "embeddings", prewarm: bool = True):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._model_lock = threading.Lock()
        self._index: Optional[faiss.Index] = None
        self._documents: List[Document] = []

        # Pre-warm the embedding model in the background so it's ready
        # by the time the user clicks "Index Documents"
        if prewarm:
            threading.Thread(target=self._get_model, daemon=True, name="embed-prewarm").start()

    def _get_model(self):
        """Lazy-load + cache the embedding model (thread-safe)."""
        with self._model_lock:
            if self._model is None:
                print(f"[VectorStore] Loading embedding model: {self.MODEL_NAME}")
                from fastembed import TextEmbedding
                self._model = TextEmbedding(self.MODEL_NAME)
                print("[VectorStore] Embedding model ready.")
        return self._model

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        model = self._get_model()
        texts = [doc.text for doc in documents]
        print(f"[VectorStore] Encoding {len(texts)} chunks...")

        # fastembed returns a generator of numpy arrays
        embeddings_gen = model.embed(texts, batch_size=64)
        embeddings = np.array(list(embeddings_gen)).astype(np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        if self._index is None:
            total = len(documents)
            if total >= 500:
                # IVFFlat — much faster search for large indexes
                n_clusters = min(int(total ** 0.5), 256)
                quantizer = faiss.IndexFlatIP(dim)
                self._index = faiss.IndexIVFFlat(
                    quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT
                )
                self._index.train(embeddings)
                self._index.nprobe = max(8, n_clusters // 8)
                print(f"[VectorStore] IVFFlat index (clusters={n_clusters})")
            else:
                self._index = faiss.IndexFlatIP(dim)
                print("[VectorStore] FlatIP index")

        self._index.add(embeddings)
        self._documents.extend(documents)
        print(f"[VectorStore] Indexed: {len(self._documents)} total chunks")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if self._index is None or not self._documents:
            return []
        model = self._get_model()
        q_emb = np.array(list(model.embed([query]))).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self._index.search(q_emb, min(top_k, len(self._documents)))
        return [
            (self._documents[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_dir / "faiss.index"))
            with open(self.index_dir / "documents.pkl", "wb") as f:
                pickle.dump(self._documents, f)
            print("[VectorStore] Index saved.")

    def load(self) -> bool:
        idx_path = self.index_dir / "faiss.index"
        doc_path = self.index_dir / "documents.pkl"
        if idx_path.exists() and doc_path.exists():
            self._index = faiss.read_index(str(idx_path))
            with open(doc_path, "rb") as f:
                self._documents = pickle.load(f)
            print(f"[VectorStore] Loaded {len(self._documents)} chunks from disk.")
            return True
        return False

    def clear(self) -> None:
        self._index = None
        self._documents = []
        for p in [self.index_dir / "faiss.index", self.index_dir / "documents.pkl"]:
            if p.exists():
                p.unlink()
        print("[VectorStore] Cleared.")

    @property
    def doc_count(self) -> int:
        return len(self._documents)
