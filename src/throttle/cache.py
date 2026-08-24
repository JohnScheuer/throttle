"""Two-tier similarity cache for LLM inference.

Tier 1 (fast): Jaccard lexical similarity
Tier 2 (slow): ONNX sentence embeddings + cosine similarity (optional)

The embedding tier is off by default and requires the 'embeddings' extra.
"""

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple, Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import torch
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    lexical_hits: int = 0
    embedding_hits: int = 0

class _OnnxEmbedder:
    """Lazy ONNX embedder for semantic similarity matching.

    Uses sentence-transformers/all-MiniLM-L6-v2 model via ONNX Runtime.
    Produces 384-dimensional L2-normalized float32 embeddings.
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not _EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not installed. "
                "Install with: pip install throttle-bench[embeddings]"
            )
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def embed(self, text: str) -> "np.ndarray":
        """Generate L2-normalized 384-dim float32 embedding."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling with attention mask
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        emb = (summed / counts)[0].detach().cpu().numpy().astype("float32")

        # L2 normalize for cosine via dot product
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

@dataclass
class _CacheEntry:
    """Internal cache entry with optional embedding."""
    prompt: str
    response_data: Any
    timestamp: float
    embedding: Optional["np.ndarray"] = None

class SimilarityCache:
    """Two-tier similarity cache for LLM inference.

    Tier 1: Jaccard lexical similarity (always enabled)
    Tier 2: ONNX semantic embeddings (optional, requires 'embeddings' extra)

    Thread-safe for concurrent use.
    """
    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        *,
        enable_embeddings: bool = False,
        embedding_threshold: float = 0.95,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_max_entries_scanned: int = 256,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= embedding_threshold <= 1.0):
            raise ValueError("embedding_threshold must be between 0.0 and 1.0")

        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.enable_embeddings = enable_embeddings
        self.embedding_threshold = embedding_threshold
        self.embedding_model_id = embedding_model_id
        self.embedding_max_entries_scanned = embedding_max_entries_scanned
        self.metrics = CacheMetrics()

        # Store maps: prompt -> (_CacheEntry)
        # Response data is scope dict when used via proxy, raw response otherwise
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._embedder: Optional[_OnnxEmbedder] = None
        self._embedding_fallback_logged = False

        # Stacked embeddings: (N, 384) array for vectorized scan
        # Keys list maintains same order as rows in stacked array
        self._embedding_matrix: Optional["np.ndarray"] = None
        self._embedding_keys: list[str] = []

        if self.enable_embeddings:
            if not _EMBEDDINGS_AVAILABLE:
                logger.warning(
                    "Embeddings requested but dependencies not installed. "
                    "Falling back to Jaccard-only matching. "
                    "Install with: pip install throttle-bench[embeddings]"
                )
                self._embedding_fallback_logged = True
                self.enable_embeddings = False
            else:
                self._embedder = _OnnxEmbedder(model_id=self.embedding_model_id)

    def _jaccard_similarity(self, prompt_a: str, prompt_b: str) -> float:
        set_a = set(prompt_a.lower().split())
        set_b = set(prompt_b.lower().split())

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _cosine_normalized(a: "np.ndarray", b: "np.ndarray") -> float:
        """Cosine similarity for L2-normalized vectors (== dot product)."""
        return float(np.dot(a, b))

    def _append_embedding_row(self, key: str, embedding: "np.ndarray"):
        """Append one embedding row to the matrix."""
        if self._embedding_matrix is None:
            self._embedding_matrix = embedding.reshape(1, -1)
            self._embedding_keys = [key]
        else:
            self._embedding_matrix = np.vstack([self._embedding_matrix, embedding])
            self._embedding_keys.append(key)

    def _remove_embedding_rows(self, keys_to_remove: set):
        """Remove rows for evicted keys without full rebuild."""
        if self._embedding_matrix is None or not keys_to_remove:
            return

        # Find indices to keep
        indices_to_keep = [
            i for i, key in enumerate(self._embedding_keys)
            if key not in keys_to_remove
        ]

        if not indices_to_keep:
            self._embedding_matrix = None
            self._embedding_keys = []
        else:
            self._embedding_matrix = self._embedding_matrix[indices_to_keep, :]
            self._embedding_keys = [self._embedding_keys[i] for i in indices_to_keep]

    def _evict_expired_unsafe(self, current_time: float):
        expired_keys = [
            k for k, entry in self._store.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        if expired_keys:
            for k in expired_keys:
                del self._store[k]
                self.metrics.evictions += 1
            # Remove evicted keys from embedding matrix
            if self.enable_embeddings:
                self._remove_embedding_rows(set(expired_keys))

    def _embed_prompt(self, prompt: str) -> "np.ndarray":
        """Generate embedding for prompt. Caller must hold lock."""
        assert self._embedder is not None
        return self._embedder.embed(prompt)

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
                self.metrics.lexical_hits += 1
                return (prompt, self._store[prompt].response_data)

            # Slow-path: Lexical match (O(N))
            for cached_prompt, entry in self._store.items():
                if self._jaccard_similarity(prompt, cached_prompt) >= self.similarity_threshold:
                    self.metrics.hits += 1
                    self.metrics.lexical_hits += 1
                    return (cached_prompt, entry.response_data)

            # Embedding tier: Semantic match (optional)
            if self.enable_embeddings and self._embedder is not None and self._store:
                query_emb = self._embed_prompt(prompt)
                best_score = -1.0
                best_key = None
                best_data = None

                # Scan last N entries (most recent in insertion order)
                entries_list = list(self._store.items())
                scan_start = max(0, len(entries_list) - self.embedding_max_entries_scanned)
                candidates = entries_list[scan_start:]

                for cached_prompt, entry in candidates:
                    if entry.embedding is None:
                        entry.embedding = self._embed_prompt(cached_prompt)

                    score = self._cosine_normalized(query_emb, entry.embedding)
                    if score > best_score:
                        best_score = score
                        best_key = cached_prompt
                        best_data = entry.response_data

                if best_score >= self.embedding_threshold and best_data is not None:
                    self.metrics.hits += 1
                    self.metrics.embedding_hits += 1
                    return (best_key, best_data)

            self.metrics.misses += 1
            return None

    def get_exact_no_metrics(self, prompt: str) -> Optional[Any]:
        """Get exact match without incrementing metrics. For internal use during cache updates."""
        with self._lock:
            if prompt in self._store:
                return self._store[prompt].response_data
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
                return (prompt, self._store[prompt].response_data)

            # Slow-path: Lexical match (O(N))
            for cached_prompt, entry in self._store.items():
                if self._jaccard_similarity(prompt, cached_prompt) >= self.similarity_threshold:
                    return (cached_prompt, entry.response_data)

            # Embedding tier: Semantic match (optional, vectorized)
            if self.enable_embeddings and self._embedder is not None and self._store:
                query_emb = self._embed_prompt(prompt)

                if self._embedding_matrix is not None and len(self._embedding_keys) > 0:
                    # Scan last N entries (window)
                    scan_start = max(0, len(self._embedding_keys) - self.embedding_max_entries_scanned)
                    scan_keys = self._embedding_keys[scan_start:]
                    scan_matrix = self._embedding_matrix[scan_start:, :]

                    # Vectorized cosine: (N,384) @ (384,) = (N,)
                    similarities = scan_matrix @ query_emb

                    # Find best match
                    best_idx = int(np.argmax(similarities))
                    best_score = float(similarities[best_idx])

                    if best_score >= self.embedding_threshold:
                        best_key = scan_keys[best_idx]
                        best_data = self._store[best_key].response_data
                        # DELIBERATE: Best match wins, then scope gate applies in proxy.
                        # If scope dict lacks requesting scope, proxy will mark miss.
                        # Do not fall through to next candidate - fail safe.
                        return (best_key, best_data)

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
                # Remove evicted key from embedding matrix
                if self.enable_embeddings:
                    self._remove_embedding_rows({oldest_key})

            # Eager embed on write if embeddings enabled
            embedding = None
            if self.enable_embeddings and self._embedder is not None:
                embedding = self._embed_prompt(prompt)

            self._store[prompt] = _CacheEntry(
                prompt=prompt,
                response_data=response_data,
                timestamp=now,
                embedding=embedding,
            )

            # Append embedding row after adding to store
            if self.enable_embeddings and embedding is not None:
                self._append_embedding_row(prompt, embedding)
