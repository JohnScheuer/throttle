"""Test embedding matrix consistency with store operations."""
import time
import pytest
import numpy as np
from throttle.cache import SimilarityCache


def _assert_matrix_consistent(cache):
    """Assert embedding matrix is consistent with store."""
    if not cache.enable_embeddings or cache._embedder is None:
        assert cache._embedding_matrix is None
        assert cache._embedding_keys == []
        return

    # Collect entries with embeddings from store
    store_keys_with_embeddings = [
        k for k, entry in cache._store.items()
        if entry.embedding is not None
    ]

    # Matrix keys should match store keys with embeddings
    assert set(cache._embedding_keys) == set(store_keys_with_embeddings)
    assert len(cache._embedding_keys) == len(store_keys_with_embeddings)

    if len(store_keys_with_embeddings) == 0:
        assert cache._embedding_matrix is None
        assert cache._embedding_keys == []
    else:
        assert cache._embedding_matrix is not None
        assert cache._embedding_matrix.shape == (len(store_keys_with_embeddings), 384)

        # Verify each row matches corresponding entry
        for i, key in enumerate(cache._embedding_keys):
            store_embedding = cache._store[key].embedding
            matrix_row = cache._embedding_matrix[i]
            assert np.allclose(matrix_row, store_embedding)


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="embeddings extra not installed"),
    reason="embeddings extra not installed"
)
def test_matrix_consistency_on_put():
    """Matrix stays consistent after put operations."""
    cache = SimilarityCache(
        ttl_seconds=3600.0,
        max_size=5,
        enable_embeddings=True,
    )

    # Initial state: empty
    _assert_matrix_consistent(cache)

    # Add first entry
    cache.put("prompt1", {"response": "data1"})
    _assert_matrix_consistent(cache)

    # Add more entries
    cache.put("prompt2", {"response": "data2"})
    cache.put("prompt3", {"response": "data3"})
    _assert_matrix_consistent(cache)


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="embeddings extra not installed"),
    reason="embeddings extra not installed"
)
def test_matrix_consistency_on_ttl_eviction():
    """Matrix stays consistent after TTL eviction."""
    cache = SimilarityCache(
        ttl_seconds=0.5,  # Short TTL
        max_size=10,
        enable_embeddings=True,
    )

    # Add entries
    cache.put("prompt1", {"response": "data1"})
    cache.put("prompt2", {"response": "data2"})
    cache.put("prompt3", {"response": "data3"})
    _assert_matrix_consistent(cache)

    # Wait for TTL expiration
    time.sleep(0.6)

    # Trigger eviction via get (calls _evict_expired_unsafe)
    cache.get("prompt4")
    _assert_matrix_consistent(cache)

    # Store should be empty after TTL eviction
    assert len(cache._store) == 0
    assert cache._embedding_matrix is None
    assert cache._embedding_keys == []


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="embeddings extra not installed"),
    reason="embeddings extra not installed"
)
def test_matrix_consistency_on_fifo_eviction():
    """Matrix stays consistent after FIFO eviction."""
    cache = SimilarityCache(
        ttl_seconds=3600.0,
        max_size=3,  # Small size to trigger FIFO
        enable_embeddings=True,
    )

    # Fill cache to max_size
    cache.put("prompt1", {"response": "data1"})
    cache.put("prompt2", {"response": "data2"})
    cache.put("prompt3", {"response": "data3"})
    _assert_matrix_consistent(cache)

    assert len(cache._store) == 3

    # Add one more to trigger FIFO eviction
    cache.put("prompt4", {"response": "data4"})
    _assert_matrix_consistent(cache)

    # Store should still be at max_size
    assert len(cache._store) == 3
    # prompt1 should have been evicted (FIFO)
    assert "prompt1" not in cache._store
    assert "prompt4" in cache._store
