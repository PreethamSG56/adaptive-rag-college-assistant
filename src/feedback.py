"""
Feedback Tracker
=================
Logs per-query metrics and adjusts retrieval behavior
based on latency sliding window.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class QueryMetrics:
    query: str
    mode: str
    retrieval_time: float
    generation_time: float
    total_time: float
    top_k_used: int
    strategy_used: str
    complexity_label: str
    timestamp: float


class FeedbackTracker:
    def __init__(self, log_file: str = "feedback_log.jsonl"):
        self.log_file = Path(log_file)
        self._metrics: List[QueryMetrics] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        self._metrics.append(QueryMetrics(**d))
                    except Exception:
                        pass

    def log(self, metrics: QueryMetrics) -> None:
        self._metrics.append(metrics)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

    def recent_latencies(self, n: int = 10) -> List[float]:
        return [m.total_time for m in self._metrics[-n:]]

    def avg_latency(self, n: int = 10) -> float:
        lats = self.recent_latencies(n)
        return sum(lats) / len(lats) if lats else 0.0

    def p50_p95(self) -> dict:
        if not self._metrics:
            return {"p50": 0, "p95": 0}
        lats = sorted(m.total_time for m in self._metrics)
        n = len(lats)
        p50 = lats[int(n * 0.5)]
        p95 = lats[min(int(n * 0.95), n - 1)]
        return {"p50": round(p50, 3), "p95": round(p95, 3)}

    def strategy_distribution(self) -> dict:
        dist: dict[str, int] = {}
        for m in self._metrics:
            dist[m.strategy_used] = dist.get(m.strategy_used, 0) + 1
        return dist

    def summary(self) -> dict:
        if not self._metrics:
            return {}
        return {
            "total_queries": len(self._metrics),
            "avg_retrieval_time": round(
                sum(m.retrieval_time for m in self._metrics) / len(self._metrics), 3
            ),
            "avg_generation_time": round(
                sum(m.generation_time for m in self._metrics) / len(self._metrics), 3
            ),
            "avg_total_time": round(self.avg_latency(), 3),
            **self.p50_p95(),
            "strategy_distribution": self.strategy_distribution(),
        }
