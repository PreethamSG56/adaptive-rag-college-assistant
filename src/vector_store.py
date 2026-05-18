"""
FAISS Vector Store
===================
Stores sentence-transformer embeddings and supports
semantic similarity search.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.ingestion import Document


class VectorStore:
    # BAAI/bge-small-en-v1.5 is extremely fast and high quality
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, index_dir: str = "embeddings"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._index: Optional[faiss.Index] = None
        self._documents: List[Document] = []

    def _get_model(self):
        if self._model is None:
            print(f"[VectorStore] Loading fast embedding model: {self.MODEL_NAME}")
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.MODEL_NAME)
        return self._model

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        model = self._get_model()
        texts = [doc.text for doc in documents]
        print(f"[VectorStore] Encoding {len(texts)} chunks using FastEmbed...")
        
        # fastembed returns a generator of numpy arrays
        embeddings_gen = model.embed(texts, batch_size=32)
        embeddings = np.array(list(embeddings_gen)).astype(np.float32)
        
        dim = embeddings.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._documents.extend(documents)
        print(f"[VectorStore] Total indexed: {len(self._documents)}")

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if self._index is None or len(self._documents) == 0:
            return []
        model = self._get_model()
        
        # embed query
        q_emb_gen = model.embed([query])
        q_emb = np.array(list(q_emb_gen)).astype(np.float32)
        faiss.normalize_L2(q_emb)
        
        scores, indices = self._index.search(q_emb, min(top_k, len(self._documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._documents[idx], float(score)))
        return results

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
            print(f"[VectorStore] Loaded {len(self._documents)} docs from disk.")
            return True
        return False

    def clear(self) -> None:
        self._index = None
        self._documents = []
        idx_path = self.index_dir / "faiss.index"
        doc_path = self.index_dir / "documents.pkl"
        for p in [idx_path, doc_path]:
            if p.exists():
                p.unlink()
        print("[VectorStore] Cleared index.")

    @property
    def doc_count(self) -> int:
        return len(self._documents)
