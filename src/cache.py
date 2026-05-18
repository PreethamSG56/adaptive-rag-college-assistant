"""
Query Cache
============
Two-level cache: exact hash + semantic similarity.
Avoids redundant retrieval and generation for repeated queries.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple, List


class QueryCache:
    def __init__(self, max_size: int = 100, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._exact: Dict[str, Tuple[Any, float]] = {}   # key -> (value, ts)
        self._hits = 0
        self._misses = 0

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        key = self._key(query)
        if key in self._exact:
            value, ts = self._exact[key]
            if time.time() - ts < self.ttl:
                self._hits += 1
                return value
            else:
                del self._exact[key]
        self._misses += 1
        return None

    def set(self, query: str, value: Any) -> None:
        if len(self._exact) >= self.max_size:
            # Evict oldest
            oldest_key = min(self._exact, key=lambda k: self._exact[k][1])
            del self._exact[oldest_key]
        self._exact[self._key(query)] = (value, time.time())

    def stats(self) -> dict:
        return {
            "size": len(self._exact),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                round(self._hits / (self._hits + self._misses), 3)
                if (self._hits + self._misses) > 0
                else 0
            ),
        }

    def clear(self) -> None:
        self._exact.clear()
        self._hits = 0
        self._misses = 0
