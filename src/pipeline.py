"""
Adaptive RAG Pipeline Orchestrator
=====================================
Ties together all components:
  Part 1: Ingestion -> VectorStore -> Generator
  Part 2: HybridRetriever (FAISS + BM25 + re-ranking)
  Part 3: AdaptiveSelector (runtime query analysis)
  Part 4: FeedbackTracker (latency logging + adjustment)
  Bonus:  QueryCache (exact hash)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env on import
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.ingestion import Document, ingest_documents
from src.vector_store import VectorStore
from src.keyword_store import KeywordStore
from src.retriever import HybridRetriever, RetrievalConfig
from src.adaptive_selector import AdaptiveSelector
from src.generator import generate_answer
from src.feedback import FeedbackTracker, QueryMetrics
from src.cache import QueryCache
from src.hallucination_guard import guard_answer, format_guard_metadata


class AdaptiveRAGPipeline:
    def __init__(
        self,
        embeddings_dir: str = "embeddings",
        feedback_log: str = "feedback_log.jsonl",
        cache_size: int = 100,
        use_cache: bool = True,
    ):
        self.vector_store = VectorStore(index_dir=embeddings_dir)
        self.keyword_store = KeywordStore()
        self.retriever = HybridRetriever(self.vector_store, self.keyword_store)
        self.selector = AdaptiveSelector(latency_threshold=8.0)
        self.feedback = FeedbackTracker(log_file=feedback_log)
        self.cache = QueryCache(max_size=cache_size)
        self.use_cache = use_cache
        self._ingested_files: List[str] = []

    # ---- ingestion -------------------------------------------------------

    def ingest(
        self,
        source: str,
        chunk_size: int = 500,
        overlap: int = 50,
        force_reload: bool = False,
    ) -> int:
        """Load documents into the pipeline. Returns number of chunks indexed."""
        # Try loading from disk first
        if not force_reload and self.vector_store.load():
            # Rebuild BM25 from the loaded vector docs
            self.keyword_store.add_documents(self.vector_store._documents)
            return self.vector_store.doc_count

        docs = ingest_documents(source, chunk_size=chunk_size, overlap=overlap)
        if not docs:
            return 0
        self.vector_store.add_documents(docs)
        self.keyword_store.add_documents(docs)
        self.vector_store.save()
        self._ingested_files.append(source)
        return len(docs)

    def ingest_documents_directly(self, documents: List[Document]) -> int:
        """Ingest pre-loaded Document objects directly."""
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        self.keyword_store.add_documents(documents)
        self.vector_store.save()
        return len(documents)

    # ---- query -----------------------------------------------------------

    def query(
        self,
        query_text: str,
        mode: str = "qa",
        force_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full adaptive RAG pipeline.
        mode: "qa" | "summarize" | "exam_questions" | "viva" | "formulas" | "explain"
        """
        t0 = time.perf_counter()

        # 1. Cache check
        cache_key = f"{mode}::{query_text}"
        if self.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached["from_cache"] = True
                return cached

        # 2. Adaptive profiling
        profile = self.selector.profile_query(query_text)
        recent_lat = self.feedback.avg_latency()
        config = self.selector.select_config(profile, recent_latency=recent_lat)

        if force_top_k is not None:
            config.top_k = force_top_k

        # 3. Retrieval
        t_ret_start = time.perf_counter()
        context_docs = self.retriever.retrieve(query_text, config)
        retrieval_time = time.perf_counter() - t_ret_start

        # 4. Generation
        t_gen_start = time.perf_counter()
        raw_answer = generate_answer(
            query_text,
            context_docs,
            mode=mode,
            model_size="large" if profile.complexity_label == "complex" else "small",
        )
        generation_time = time.perf_counter() - t_gen_start

        # 4b. Hallucination Guard — verify answer against context
        guard_result = guard_answer(
            raw_answer,
            context_docs,
            grounding_threshold=0.12,
            pass_ratio=0.50,
            partial_ratio=0.25,
        )
        answer = guard_result.grounded_answer

        total_time = time.perf_counter() - t0

        # 5. Feedback logging
        metrics = QueryMetrics(
            query=query_text,
            mode=mode,
            retrieval_time=round(retrieval_time, 4),
            generation_time=round(generation_time, 4),
            total_time=round(total_time, 4),
            top_k_used=config.top_k,
            strategy_used=config.strategy,
            complexity_label=profile.complexity_label,
            timestamp=time.time(),
        )
        self.feedback.log(metrics)
        self.selector.record_latency(total_time)

        result = {
            "query": query_text,
            "answer": answer,
            "mode": mode,
            "sources": [
                {"source": doc.source, "chunk_id": doc.chunk_id, "score": round(score, 4)}
                for doc, score in context_docs
            ],
            "retrieval_config": {
                "top_k": config.top_k,
                "strategy": config.strategy,
                "rerank": config.rerank,
            },
            "query_profile": {
                "complexity": profile.complexity_label,
                "word_count": profile.word_count,
                "study_intent": profile.has_study_intent,
            },
            "timings": {
                "retrieval_s": round(retrieval_time, 4),
                "generation_s": round(generation_time, 4),
                "total_s": round(total_time, 4),
            },
            # Hallucination guard metadata
            "grounding": {
                "verdict": guard_result.verdict,
                "confidence": guard_result.overall_confidence,
                "grounded_ratio": guard_result.grounded_sentence_ratio,
                "ocr_warning": guard_result.ocr_quality_warning,
                "ungrounded_count": len(guard_result.ungrounded_sentences),
                "summary": format_guard_metadata(guard_result),
            },
            "from_cache": False,
        }

        # Store in cache
        if self.use_cache:
            self.cache.set(cache_key, result)

        return result

    # ---- stats -----------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "indexed_docs": self.vector_store.doc_count,
            "cache": self.cache.stats(),
            "feedback": self.feedback.summary(),
        }

    def clear_all(self) -> None:
        self.vector_store.clear()
        self.keyword_store.clear()
        self.cache.clear()
        print("[Pipeline] Cleared all indexes and cache.")
