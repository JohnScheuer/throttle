import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple, Any

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

class SimilarityCache:
    """
    An In-Memory Similarity-based Cache for LLM inferences.
    Uses lexical Jaccard similarity for matching and supports TTL and max-size eviction.
    Thread-safe for concurrent benchmarking.
    """
    def __init__(
        self, 
        ttl_seconds: float = 3600.0, 
        max_size: int = 1000, 
        similarity_threshold: float = 0.85
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.metrics = CacheMetrics()
        
        # Store maps: prompt -> (response_data, timestamp)
        # We store structured response data to preserve token usage/status if needed
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = Lock()

    def _jaccard_similarity(self, prompt_a: str, prompt_b: str) -> float:
        set_a = set(prompt_a.lower().split())
        set_b = set(prompt_b.lower().split())
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def _evict_expired_unsafe(self, current_time: float):
        expired_keys = [
            k for k, (_, ts) in self._store.items() 
            if current_time - ts > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._store[k]
            self.metrics.evictions += 1

    def get(self, prompt: str) -> Optional[Any]:
        """Retrieves structured response data if an exact or similarity match is found."""
        result = self.get_with_key(prompt)
        return result[1] if result else None

    def get_with_key(self, prompt: str) -> Optional[tuple[str, Any]]:
        """Retrieves (canonical_key, response_data) if an exact or similarity match is found.

        Returns the matched cache key along with the value, allowing callers to update
        the same entry when adding scope variants.
        """
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            # Fast-path: Exact match (O(1))
            if prompt in self._store:
                self.metrics.hits += 1
                return (prompt, self._store[prompt][0])

            # Slow-path: Lexical match (O(N))
            for cached_prompt, (response_data, _) in self._store.items():
                if self._jaccard_similarity(prompt, cached_prompt) >= self.similarity_threshold:
                    self.metrics.hits += 1
                    return (cached_prompt, response_data)

            self.metrics.misses += 1
            return None

    def get_exact_no_metrics(self, prompt: str) -> Optional[Any]:
        """Get exact match without incrementing metrics. For internal use during cache updates."""
        with self._lock:
            if prompt in self._store:
                return self._store[prompt][0]
            return None

    def get_with_key_no_metrics(self, prompt: str) -> Optional[tuple[str, Any]]:
        """Retrieves (canonical_key, response_data) without incrementing metrics.

        Returns the matched cache key along with the value, for scope validation
        before committing to a cache hit or miss in metrics.
        """
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            # Fast-path: Exact match (O(1))
            if prompt in self._store:
                return (prompt, self._store[prompt][0])

            # Slow-path: Lexical match (O(N))
            for cached_prompt, (response_data, _) in self._store.items():
                if self._jaccard_similarity(prompt, cached_prompt) >= self.similarity_threshold:
                    return (cached_prompt, response_data)

            return None

    def put(self, prompt: str, response_data: Any):
        """Stores structured response data in the cache."""
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)
            
            # Evict oldest (FIFO) if we hit the size limit
            if len(self._store) >= self.max_size and prompt not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
                self.metrics.evictions += 1
                
            self._store[prompt] = (response_data, now)
