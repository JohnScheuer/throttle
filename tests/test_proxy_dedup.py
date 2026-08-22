"""Tests for proxy in-flight request deduplication without requiring live backend."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from throttle.proxy import ProxyServer


class MockBackend:
    """Mock inference backend for testing deduplication behavior."""

    def __init__(self, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds
        self.request_count = 0
        self.requests_received: List[Dict] = []
        self._lock = asyncio.Lock()
        self._should_error = False
        self._should_timeout = False
        self._timeout_duration = 30.0

    def configure_error(self):
        """Configure backend to return errors."""
        self._should_error = True

    def configure_timeout(self, duration: float):
        """Configure backend to timeout."""
        self._should_timeout = True
        self._timeout_duration = duration

    async def handle_request(self, request: Request):
        """Handle a chat completion request."""
        async with self._lock:
            self.request_count += 1
            request_num = self.request_count

        body = await request.json()
        self.requests_received.append({
            "request_num": request_num,
            "body": body,
            "timestamp": time.time(),
        })

        # Artificial delay to make concurrency observable
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # Simulate timeout
        if self._should_timeout:
            await asyncio.sleep(self._timeout_duration)

        # Simulate error
        if self._should_error:
            return JSONResponse(
                status_code=500,
                content={"error": "Mock backend error"}
            )

        # Return unique identifiable response
        return JSONResponse({
            "id": f"mock-response-{request_num}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response #{request_num}"
                },
                "finish_reason": "stop"
            }]
        })


@pytest.fixture
async def mock_backend_server():
    """Fixture providing a mock backend server."""
    import uvicorn

    mock = MockBackend(delay_seconds=0.1)  # 100ms delay makes concurrency observable
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        return await mock.handle_request(request)

    config = uvicorn.Config(app, host="127.0.0.1", port=58090, log_level="error")
    server = uvicorn.Server(config)

    # Start server in background
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for server to start

    yield mock

    # Shutdown
    server.should_exit = True
    await task


@pytest.fixture
async def proxy_to_mock(mock_backend_server):
    """Fixture providing a proxy pointing to mock backend."""
    import uvicorn

    proxy = ProxyServer(
        backend_url="http://127.0.0.1:58090",
        enable_cache=True,
        cache_ttl_seconds=3600.0,
        cache_max_size=100,
        cache_similarity_threshold=0.85,
    )

    app = proxy.app

    @app.on_event("startup")
    async def startup():
        await proxy.startup()

    @app.on_event("shutdown")
    async def shutdown():
        await proxy.shutdown()

    config = uvicorn.Config(app, host="127.0.0.1", port=58091, log_level="error")
    server = uvicorn.Server(config)

    # Start server in background
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for proxy to start

    yield proxy

    # Shutdown
    server.should_exit = True
    await task


async def test_five_concurrent_identical_requests_one_backend_call(proxy_to_mock, mock_backend_server):
    """5 concurrent identical requests result in exactly 1 backend call and 5 byte-identical responses."""
    mock = mock_backend_server
    proxy = proxy_to_mock

    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": False,
    }

    async def send_request(client_id: int):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json=payload,
            )
            return client_id, response.json()

    # Send 5 concurrent identical requests
    tasks = [send_request(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    # Assert exactly 1 backend call (deduplication worked)
    assert mock.request_count == 1, f"Expected 1 backend call, got {mock.request_count}"

    # Assert backend_calls counter matches mock's count
    async with httpx.AsyncClient() as health_client:
        health = await health_client.get("http://127.0.0.1:58091/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == 1, f"Counter shows {stats['backend_calls']}, mock received {mock.request_count}"
        assert stats["misses"] == 5, f"Expected 5 cache misses (all 5 checked cache)"


async def test_ten_concurrent_paraphrases_current_behavior(proxy_to_mock, mock_backend_server):
    """10 concurrent paraphrases: assert CURRENT behavior (each reaches backend).

    This test documents current behavior where paraphrases are NOT deduplicated.
    When Promise Cache lands, this test should be updated to expect deduplication.
    """
    mock = mock_backend_server

    # 10 prompts: 2 groups of paraphrases
    prompts = [
        "How do I sort a Python list?",           # 0
        "How can I sort a list in Python?",       # 1 - paraphrase of 0
        "What's the way to sort a Python list?",  # 2 - paraphrase of 0
        "How to sort lists in Python?",           # 3 - paraphrase of 0
        "What is the capital of France?",         # 4
        "What's France's capital city?",          # 5 - paraphrase of 4
        "Tell me the capital of France",          # 6 - paraphrase of 4
        "How do I sort a Python list?",           # 7 - exact dup of 0
        "What is the capital of France?",         # 8 - exact dup of 4
        "Which city is the capital of France?",   # 9 - paraphrase of 4
    ]

    async def send_request(idx: int, prompt: str):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            return idx, response.json()

    tasks = [send_request(i, prompts[i]) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # CURRENT BEHAVIOR: Paraphrases are NOT deduplicated, only exact matches
    # Expected: 8 backend calls (10 prompts - 2 exact dups)
    # When Promise Cache lands, expect: 2 backend calls (one per semantic cluster)
    expected_backend_calls = 8
    assert mock.request_count == expected_backend_calls, \
        f"CURRENT behavior: Expected {expected_backend_calls} backend calls, got {mock.request_count}"

    # Assert backend_calls counter matches mock
    async with httpx.AsyncClient() as health_client:
        health = await health_client.get("http://127.0.0.1:58091/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == expected_backend_calls


async def test_backend_error_propagates_to_all_waiters(proxy_to_mock, mock_backend_server):
    """Backend error during in-flight request returns error to all concurrent waiters."""
    mock = mock_backend_server
    mock.configure_error()

    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Test error propagation"}],
        "stream": False,
    }

    async def send_request(client_id: int):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json=payload,
            )
            return client_id, response.status_code, response.text

    # Send 3 concurrent requests - bounded to 5 seconds to catch hangs
    start_time = time.time()
    tasks = [send_request(i) for i in range(3)]

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Test hung - concurrent requests did not complete within 5 seconds")

    elapsed = time.time() - start_time

    # Assert exactly 1 backend call (deduplication was in effect when error occurred)
    assert mock.request_count == 1, f"Expected 1 backend call, got {mock.request_count}"

    # Assert all 3 responses are errors (500 status)
    assert len(results) == 3, "Should have 3 results"
    for client_id, status_code, response_text in results:
        assert status_code == 500, f"Client {client_id} got status {status_code}, expected 500"
        assert "error" in response_text.lower(), f"Client {client_id} response missing error indication"

    # Assert completion was reasonably fast (not hung)
    assert elapsed < 3.0, f"Requests took {elapsed:.2f}s, too slow (likely hung)"

    # Assert subsequent request is NOT served from cache (failed entry was evicted)
    mock.request_count = 0  # Reset mock counter
    mock._should_error = False  # Disable error for next request

    async with httpx.AsyncClient(timeout=10.0) as client:
        retry_response = await client.post(
            "http://127.0.0.1:58091/v1/chat/completions",
            json=payload,
        )
        assert retry_response.status_code == 200, "Retry should succeed"
        # Should have reached backend again (not served stale error from cache)
        assert mock.request_count == 1, "Retry should reach backend, not serve cached error"


async def test_backend_timeout_does_not_hang_waiters():
    """Backend timeout during in-flight request propagates to all waiters and is not cached."""
    import uvicorn

    # Create mock backend with delay that exceeds proxy timeout
    mock = MockBackend(delay_seconds=0.1)
    mock.configure_timeout(duration=2.0)  # 2s delay will exceed proxy's 1s timeout

    app = FastAPI()
    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        return await mock.handle_request(request)

    config = uvicorn.Config(app, host="127.0.0.1", port=58092, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for server to start

    # Create proxy with short timeout for testing
    proxy = ProxyServer(
        backend_url="http://127.0.0.1:58092",
        enable_cache=True,
        cache_ttl_seconds=3600.0,
        cache_max_size=100,
        cache_similarity_threshold=0.85,
    )
    proxy._client = httpx.AsyncClient(timeout=1.0)  # 1s timeout for testing

    proxy_app = proxy.app
    @proxy_app.on_event("startup")
    async def startup():
        pass  # Client already created

    @proxy_app.on_event("shutdown")
    async def shutdown():
        await proxy._client.aclose()

    proxy_config = uvicorn.Config(proxy_app, host="127.0.0.1", port=58093, log_level="error")
    proxy_server = uvicorn.Server(proxy_config)
    proxy_task = asyncio.create_task(proxy_server.serve())
    await asyncio.sleep(1)

    try:
        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Test timeout"}],
            "stream": False,
        }

        async def send_request(client_id: int):
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.post(
                        "http://127.0.0.1:58093/v1/chat/completions",
                        json=payload,
                    )
                    # Proxy returns 504 on backend timeout
                    if response.status_code == 504:
                        return client_id, "timeout", 504
                    return client_id, "success", response.status_code
                except httpx.ReadTimeout:
                    return client_id, "timeout", None
                except Exception as e:
                    return client_id, "error", str(type(e).__name__)

        # Send 3 concurrent requests - proxy backend will timeout
        start_time = time.time()
        tasks = [send_request(i) for i in range(3)]

        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("Test hung - concurrent requests did not complete within 10 seconds")

        elapsed = time.time() - start_time

        # Assert exactly 1 backend call (deduplication was in effect)
        assert mock.request_count == 1, f"Expected 1 backend call, got {mock.request_count}"

        # Assert all 3 responses indicate error (proxy timeout propagated)
        assert len(results) == 3, "Should have 3 results"
        error_count = sum(1 for _, status, _ in results if status in ["timeout", "error"])
        assert error_count == 3, f"Expected all 3 to get error/timeout, got {error_count}"

        # Assert completion was reasonably fast (all failed together)
        assert elapsed < 5.0, f"Requests took {elapsed:.2f}s, too slow"

        # Assert timeout error was NOT cached (subsequent request reaches backend)
        mock.request_count = 0
        mock._should_timeout = False  # Disable timeout for next request

        async with httpx.AsyncClient(timeout=10.0) as client:
            retry_response = await client.post(
                "http://127.0.0.1:58093/v1/chat/completions",
                json=payload,
            )
            assert retry_response.status_code == 200, "Retry should succeed"
            # Should have reached backend (error was not cached)
            assert mock.request_count == 1, "Retry should reach backend, timeout not cached"

    finally:
        # Shutdown servers
        server.should_exit = True
        proxy_server.should_exit = True
        await task
        await proxy_task


async def test_empty_choices_not_cached(proxy_to_mock, mock_backend_server):
    """Backend response with no choices is returned but not cached."""
    mock = mock_backend_server

    # Temporarily override mock to return empty choices
    original_handle = mock.handle_request

    async def empty_choices_handler(request: Request):
        async with mock._lock:
            mock.request_count += 1

        # Return 200 but with no choices array
        return JSONResponse({
            "id": "empty-response",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": []  # Empty choices
        })

    mock.handle_request = empty_choices_handler

    try:
        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Test empty choices"}],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            # First request gets empty choices
            response1 = await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json=payload,
            )
            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["choices"] == [], "Should return empty choices"
            assert mock.request_count == 1

            # Second identical request should reach backend (not cached)
            response2 = await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json=payload,
            )
            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["choices"] == [], "Should return empty choices"

            # Should have made 2 backend calls (empty response not cached)
            assert mock.request_count == 2, \
                f"Expected 2 backend calls (empty not cached), got {mock.request_count}"

    finally:
        # Restore original handler
        mock.handle_request = original_handle


async def test_sequential_identical_requests_hit_cache(proxy_to_mock, mock_backend_server):
    """Sequential identical requests hit cache normally, confirming dedup didn't break ordinary path."""
    mock = mock_backend_server

    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Sequential test"}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # First request - cache miss
        response1 = await client.post(
            "http://127.0.0.1:58091/v1/chat/completions",
            json=payload,
        )
        assert response1.status_code == 200
        content1 = response1.json()["choices"][0]["message"]["content"]

        # Small delay
        await asyncio.sleep(0.2)

        # Second request - should be cache hit
        response2 = await client.post(
            "http://127.0.0.1:58091/v1/chat/completions",
            json=payload,
        )
        assert response2.status_code == 200

        # Only 1 backend call should have been made (second request was cache hit)
        assert mock.request_count == 1, f"Expected 1 backend call, got {mock.request_count}"

        # Check cache stats
        health = await client.get("http://127.0.0.1:58091/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == 1
        assert stats["hits"] == 1, "Second request should be cache hit"
        assert stats["misses"] == 1, "First request should be cache miss"


async def test_backend_calls_counter_matches_mock_count(proxy_to_mock, mock_backend_server):
    """backend_calls counter must match mock backend's actual received count."""
    mock = mock_backend_server

    # Send various requests
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 3 different prompts = 3 backend calls
        for i in range(3):
            await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": f"Prompt {i}"}],
                    "stream": False,
                },
            )

        # Check agreement
        health = await client.get("http://127.0.0.1:58091/health")
        stats = health.json()["cache_stats"]

        assert stats["backend_calls"] == mock.request_count, \
            f"Counter disagreement: counter={stats['backend_calls']}, mock={mock.request_count}"
        assert stats["backend_calls"] == 3


async def test_concurrent_requests_with_mixed_prompts(proxy_to_mock, mock_backend_server):
    """Mixed concurrent requests: some identical, some unique."""
    mock = mock_backend_server

    # 8 requests: 3 identical, 3 identical, 2 unique
    prompts = [
        "Prompt A",  # 0
        "Prompt A",  # 1 - dup
        "Prompt A",  # 2 - dup
        "Prompt B",  # 3
        "Prompt B",  # 4 - dup
        "Prompt B",  # 5 - dup
        "Prompt C",  # 6 - unique
        "Prompt D",  # 7 - unique
    ]

    async def send_request(idx: int):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://127.0.0.1:58091/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": prompts[idx]}],
                    "stream": False,
                },
            )
            return idx, response.json()

    tasks = [send_request(i) for i in range(8)]
    results = await asyncio.gather(*tasks)

    # Expected: 4 backend calls (A, B, C, D) due to deduplication
    assert mock.request_count == 4, f"Expected 4 backend calls, got {mock.request_count}"

    # Counter should match
    async with httpx.AsyncClient() as health_client:
        health = await health_client.get("http://127.0.0.1:58091/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == 4
        assert stats["backend_calls"] == mock.request_count
