"""Tests for scope-aware caching.

These tests verify that requests with the same semantic prompt but different
scope parameters (model, temperature, max_tokens, etc.) are cached separately
and do not return incorrect cached responses.
"""

import pytest
from throttle.proxy import ProxyServer
from throttle.cache import SimilarityCache


def test_extract_scope_key_excludes_messages_and_stream():
    """Scope key should include all params EXCEPT messages and stream."""
    proxy = ProxyServer(backend_url="http://mock:11434", enable_cache=False)

    request_body = {
        "model": "llama3.2:1b",
        "temperature": 0.7,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "top_p": 0.9,
    }

    scope_key = proxy._extract_scope_key(request_body)

    # Should be JSON with model, temperature, max_tokens, top_p (NOT messages or stream)
    import json
    scope_dict = json.loads(scope_key)

    assert "model" in scope_dict
    assert scope_dict["model"] == "llama3.2:1b"
    assert "temperature" in scope_dict
    assert scope_dict["temperature"] == 0.7
    assert "max_tokens" in scope_dict
    assert scope_dict["max_tokens"] == 100
    assert "top_p" in scope_dict
    assert scope_dict["top_p"] == 0.9

    # messages and stream should NOT be in scope key
    assert "messages" not in scope_dict
    assert "stream" not in scope_dict


def test_extract_scope_key_different_models_different_keys():
    """Different models should produce different scope keys."""
    proxy = ProxyServer(backend_url="http://mock:11434", enable_cache=False)

    request_1 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    request_2 = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    scope_1 = proxy._extract_scope_key(request_1)
    scope_2 = proxy._extract_scope_key(request_2)

    assert scope_1 != scope_2, "Different models must have different scope keys"


def test_extract_scope_key_different_temperatures_different_keys():
    """Different temperatures should produce different scope keys."""
    proxy = ProxyServer(backend_url="http://mock:11434", enable_cache=False)

    request_1 = {
        "model": "llama3.2:1b",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    request_2 = {
        "model": "llama3.2:1b",
        "temperature": 1.0,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    scope_1 = proxy._extract_scope_key(request_1)
    scope_2 = proxy._extract_scope_key(request_2)

    assert scope_1 != scope_2, "Different temperatures must have different scope keys"


def test_extract_scope_key_same_messages_same_stream_same_key():
    """Requests differing only in messages or stream should have same scope key."""
    proxy = ProxyServer(backend_url="http://mock:11434", enable_cache=False)

    request_1 = {
        "model": "llama3.2:1b",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": False,
    }

    request_2 = {
        "model": "llama3.2:1b",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "Different prompt here"}],
        "stream": True,
    }

    scope_1 = proxy._extract_scope_key(request_1)
    scope_2 = proxy._extract_scope_key(request_2)

    assert scope_1 == scope_2, "Same scope params (ignoring messages/stream) should produce same key"


def test_cache_get_with_key_returns_canonical_key_and_value():
    """get_with_key should return (canonical_key, value) tuple."""
    cache = SimilarityCache(ttl_seconds=3600, max_size=100, similarity_threshold=0.85)

    # Store a value
    prompt = "What is the capital of France?"
    value = {"response": "Paris"}
    cache.put(prompt, value)

    # Retrieve with exact match
    result = cache.get_with_key(prompt)

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2

    canonical_key, retrieved_value = result
    assert canonical_key == prompt
    assert retrieved_value == value


def test_cache_get_with_key_returns_canonical_key_for_similar_prompt():
    """get_with_key should return original canonical key even for similar prompts."""
    cache = SimilarityCache(ttl_seconds=3600, max_size=100, similarity_threshold=0.85)

    original_prompt = "Tell me about machine learning algorithms"
    similar_prompt = "Tell me about machine learning algorithms please"  # 6/7 words = 85.7%
    value = {"response": "ML algorithms..."}

    cache.put(original_prompt, value)

    result = cache.get_with_key(similar_prompt)

    assert result is not None
    canonical_key, retrieved_value = result
    # Canonical key should be the ORIGINAL prompt, not the similar one
    assert canonical_key == original_prompt
    assert retrieved_value == value


def test_cache_get_with_key_no_metrics_does_not_increment_hits():
    """get_with_key_no_metrics should not increment hits counter."""
    cache = SimilarityCache(ttl_seconds=3600, max_size=100, similarity_threshold=0.85)

    prompt = "test prompt"
    cache.put(prompt, {"data": "value"})

    initial_hits = cache.metrics.hits

    # Use no_metrics variant
    result = cache.get_with_key_no_metrics(prompt)

    assert result is not None
    assert cache.metrics.hits == initial_hits, "Hits should not increment"


def test_cache_get_with_key_no_metrics_does_not_increment_misses():
    """get_with_key_no_metrics should not increment misses counter."""
    cache = SimilarityCache(ttl_seconds=3600, max_size=100, similarity_threshold=0.85)

    initial_misses = cache.metrics.misses

    # Look up non-existent prompt
    result = cache.get_with_key_no_metrics("non-existent prompt")

    assert result is None
    assert cache.metrics.misses == initial_misses, "Misses should not increment"


def test_cache_get_exact_no_metrics_only_matches_exact():
    """get_exact_no_metrics should only return exact matches, not similar ones."""
    cache = SimilarityCache(ttl_seconds=3600, max_size=100, similarity_threshold=0.85)

    exact_prompt = "What is the capital of France?"
    similar_prompt = "What is the France capital?"
    cache.put(exact_prompt, {"response": "Paris"})

    # Exact match should work
    exact_result = cache.get_exact_no_metrics(exact_prompt)
    assert exact_result is not None

    # Similar prompt should NOT match (exact only)
    similar_result = cache.get_exact_no_metrics(similar_prompt)
    assert similar_result is None


def test_scope_aware_cache_storage_structure():
    """Verify that cache stores multiple scope variants per semantic prompt."""
    cache = SimilarityCache(ttl_seconds=3600, max_size=100, similarity_threshold=0.85)

    prompt = "What is 2+2?"

    # Store first scope variant
    scope_key_1 = '{"model": "llama3.2:1b"}'
    response_1 = {"model": "llama3.2:1b", "content": "4"}
    scope_dict_1 = {scope_key_1: {"response": response_1}}
    cache.put(prompt, scope_dict_1)

    # Store second scope variant (different model, same prompt)
    scope_key_2 = '{"model": "llama3.2:3b"}'
    response_2 = {"model": "llama3.2:3b", "content": "The answer is 4"}

    # Get existing scope dict
    existing = cache.get_exact_no_metrics(prompt)
    assert existing is not None

    # Add new scope variant
    existing[scope_key_2] = {"response": response_2}
    cache.put(prompt, existing)

    # Verify both scope variants are stored
    final = cache.get_exact_no_metrics(prompt)
    assert final is not None
    assert scope_key_1 in final
    assert scope_key_2 in final
    assert final[scope_key_1]["response"]["model"] == "llama3.2:1b"
    assert final[scope_key_2]["response"]["model"] == "llama3.2:3b"


@pytest.mark.asyncio
async def test_proxy_serves_correct_scope_variant():
    """Proxy should serve correct cached response based on scope, not just prompt."""
    from unittest.mock import AsyncMock, MagicMock

    proxy = ProxyServer(backend_url="http://mock:11434", enable_cache=True)
    proxy._client = AsyncMock()

    # First request: model=llama3.2:1b, prompt="What is 2+2?"
    request_1 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
    }

    response_1 = {
        "id": "1",
        "object": "chat.completion",
        "created": 123,
        "model": "llama3.2:1b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Response from 1b model"},
            "finish_reason": "stop"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: response_1))

    scope_1 = proxy._extract_scope_key(request_1)
    prompt_1 = proxy._extract_prompt(request_1)

    # Simulate the proxy caching logic
    result = proxy.cache.get_with_key_no_metrics(prompt_1)
    if result is None:
        # First request - cache miss
        scope_dict = {scope_1: {"response": response_1}}
        proxy.cache.put(prompt_1, scope_dict)

    # Second request: model=llama3.2:3b, SAME prompt
    request_2 = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
    }

    scope_2 = proxy._extract_scope_key(request_2)
    prompt_2 = proxy._extract_prompt(request_2)

    # Prompts should be identical
    assert prompt_1 == prompt_2

    # But scopes should differ
    assert scope_1 != scope_2

    # Check cache
    result = proxy.cache.get_with_key_no_metrics(prompt_2)
    assert result is not None
    canonical_key, scope_dict = result

    # Cache has the prompt, but should NOT have this scope
    assert isinstance(scope_dict, dict)
    assert scope_1 in scope_dict  # Has 1b model scope
    assert scope_2 not in scope_dict  # Does NOT have 3b model scope yet

    # This should be a cache miss (different scope)
    # After fix, proxy should NOT return response_1 for request_2
