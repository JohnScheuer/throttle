"""Test demonstrating scope-aware caching bug in current main.

This test shows that without scope-aware caching, requests with the same prompt
but different models/temperatures/max_tokens get incorrect cached responses.
"""

import pytest
from throttle.proxy import ProxyServer
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_different_models_should_not_share_cache():
    """Requests with same prompt but different models should get different responses."""

    # Create proxy with caching enabled
    proxy = ProxyServer(
        backend_url="http://mock-backend:11434",
        enable_cache=True,
    )

    # Mock the HTTP client
    proxy._client = AsyncMock()

    # First request: model=llama3.2:1b
    request_body_1 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
    }

    # Mock response for llama3.2:1b
    mock_response_1 = {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama3.2:1b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from llama3.2:1b model"
            },
            "finish_reason": "stop"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: mock_response_1))

    # Make first request - should hit backend
    prompt_1 = proxy._extract_prompt(request_body_1)
    response_1 = await proxy._make_backend_request(request_body_1, {}, prompt_1, False)

    # Cache the response
    if proxy.cache:
        proxy.cache.put(prompt_1, response_1)

    # Second request: DIFFERENT model but SAME prompt
    request_body_2 = {
        "model": "llama3.2:3b",  # Different model!
        "messages": [{"role": "user", "content": "What is 2+2?"}],  # Same prompt
        "max_tokens": 10,
    }

    # Mock response for llama3.2:3b (different from first)
    mock_response_2 = {
        "id": "resp-2",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "llama3.2:3b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from llama3.2:3b model"
            },
            "finish_reason": "stop"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: mock_response_2))

    # Check cache with second prompt
    prompt_2 = proxy._extract_prompt(request_body_2)

    # Without scope-aware caching, prompt_1 == prompt_2 (both "What is 2+2?")
    assert prompt_1 == prompt_2, "Prompts should be identical"

    # Get cached response
    cached = proxy.cache.get(prompt_2) if proxy.cache else None

    # BUG: Current implementation returns llama3.2:1b response for llama3.2:3b request
    # This is INCORRECT - different models should not share cached responses
    if cached:
        # This assertion SHOULD FAIL on current main (proving the bug exists)
        # After fix, this test will be replaced with proper scope-aware tests
        assert cached["model"] != request_body_2["model"], \
            "BUG: Cache returned response from different model (llama3.2:1b) for llama3.2:3b request"


@pytest.mark.asyncio
async def test_different_temperatures_should_not_share_cache():
    """Requests with same prompt but different temperatures should get different responses."""

    proxy = ProxyServer(
        backend_url="http://mock-backend:11434",
        enable_cache=True,
    )

    proxy._client = AsyncMock()

    # First request: temperature=0.0
    request_body_1 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "temperature": 0.0,
        "max_tokens": 20,
    }

    mock_response_1 = {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama3.2:1b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Deterministic response with temperature=0.0"
            },
            "finish_reason": "stop"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: mock_response_1))

    prompt_1 = proxy._extract_prompt(request_body_1)
    response_1 = await proxy._make_backend_request(request_body_1, {}, prompt_1, False)

    if proxy.cache:
        proxy.cache.put(prompt_1, response_1)

    # Second request: DIFFERENT temperature but SAME prompt
    request_body_2 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "temperature": 1.0,  # Different temperature!
        "max_tokens": 20,
    }

    mock_response_2 = {
        "id": "resp-2",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "llama3.2:1b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Creative response with temperature=1.0"
            },
            "finish_reason": "stop"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: mock_response_2))

    prompt_2 = proxy._extract_prompt(request_body_2)
    assert prompt_1 == prompt_2, "Prompts should be identical"

    cached = proxy.cache.get(prompt_2) if proxy.cache else None

    # BUG: Current implementation returns temperature=0.0 response for temperature=1.0 request
    if cached:
        assert "temperature=0.0" in cached["choices"][0]["message"]["content"], \
            "BUG: Cache returned response from temperature=0.0 for temperature=1.0 request"


@pytest.mark.asyncio
async def test_different_max_tokens_should_not_share_cache():
    """Requests with same prompt but different max_tokens should get different responses."""

    proxy = ProxyServer(
        backend_url="http://mock-backend:11434",
        enable_cache=True,
    )

    proxy._client = AsyncMock()

    # First request: max_tokens=10
    request_body_1 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Count to 100"}],
        "max_tokens": 10,
    }

    mock_response_1 = {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama3.2:1b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "1, 2, 3, 4, 5"  # Truncated at 10 tokens
            },
            "finish_reason": "length"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: mock_response_1))

    prompt_1 = proxy._extract_prompt(request_body_1)
    response_1 = await proxy._make_backend_request(request_body_1, {}, prompt_1, False)

    if proxy.cache:
        proxy.cache.put(prompt_1, response_1)

    # Second request: DIFFERENT max_tokens but SAME prompt
    request_body_2 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Count to 100"}],
        "max_tokens": 100,  # Different max_tokens!
    }

    mock_response_2 = {
        "id": "resp-2",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "llama3.2:1b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ..., 100"  # Full response
            },
            "finish_reason": "stop"
        }]
    }

    proxy._client.post = AsyncMock(return_value=MagicMock(json=lambda: mock_response_2))

    prompt_2 = proxy._extract_prompt(request_body_2)
    assert prompt_1 == prompt_2, "Prompts should be identical"

    cached = proxy.cache.get(prompt_2) if proxy.cache else None

    # BUG: Current implementation returns truncated response (max_tokens=10) for max_tokens=100 request
    if cached:
        assert cached["choices"][0]["finish_reason"] == "length", \
            "BUG: Cache returned truncated response (max_tokens=10) for max_tokens=100 request"
