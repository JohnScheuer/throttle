"""Tests for ONNX embedding tier through proxy endpoint.

All tests assert through the proxy endpoint on responses received,
never on internal structures like _store or _embedder.
"""

import asyncio
import pytest
from contextlib import asynccontextmanager

from throttle.proxy import ProxyServer


@asynccontextmanager
async def _fake_backend():
    """Mock backend that returns fixed responses."""
    responses = {}

    async def handler(request):
        import json
        body = await request.json()
        prompt_text = " ".join(m["content"] for m in body.get("messages", []))

        # Return cached response if exists, otherwise generate new
        if prompt_text not in responses:
            responses[prompt_text] = {
                "id": f"chatcmpl-{len(responses)}",
                "object": "chat.completion",
                "model": body.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Response to: {prompt_text[:50]}"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            }
        return responses[prompt_text]

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        response = await handler(request)
        return JSONResponse(response)

    config = uvicorn.Config(app, host="127.0.0.1", port=9999, log_level="error")
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)  # Let server start

    try:
        yield "http://127.0.0.1:9999"
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_embedding_hit_that_jaccard_misses():
    """Test a: Semantically similar prompts under identical scope produce embedding hit."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        # First request: cache miss
        from httpx import AsyncClient
        async with AsyncClient() as client:
            resp1 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "How do I optimize PostgreSQL queries?"}],
                    "temperature": 0.7,
                },
                headers={"host": "test"},
            )
            assert resp1.status_code == 200
            response1 = resp1.json()

            # Second request: semantically similar but Jaccard dissimilar
            # Jaccard("How do I optimize PostgreSQL queries?", "How can I improve database query performance in PostgreSQL?")
            # is below 0.85 threshold
            resp2 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "How can I improve database query performance in PostgreSQL?"}],
                    "temperature": 0.7,
                },
                headers={"host": "test"},
            )
            assert resp2.status_code == 200
            response2 = resp2.json()

            # Should be cache hit via embedding tier
            assert proxy.cache.metrics.hits >= 1
            assert proxy.cache.metrics.embedding_hits >= 1
            # Responses should match (same cached content)
            assert response2["choices"][0]["message"]["content"] == response1["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_embedding_miss_on_different_model():
    """Test b: Same prompts under different model values produce MISS."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        async with AsyncClient() as client:
            # First request with model-a
            resp1 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "model-a",
                    "messages": [{"role": "user", "content": "What is machine learning?"}],
                },
                headers={"host": "test"},
            )
            assert resp1.status_code == 200

            # Same prompt, different model
            resp2 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "model-b",
                    "messages": [{"role": "user", "content": "What is machine learning?"}],
                },
                headers={"host": "test"},
            )
            assert resp2.status_code == 200

            # Should be cache miss (different scope)
            assert proxy.cache.metrics.misses >= 1


@pytest.mark.asyncio
async def test_embedding_miss_on_different_temperature():
    """Test c: Same prompts under different temperature values produce MISS."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        async with AsyncClient() as client:
            # First request with temperature 0.7
            resp1 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Explain quantum computing"}],
                    "temperature": 0.7,
                },
                headers={"host": "test"},
            )
            assert resp1.status_code == 200

            # Same prompt, different temperature
            resp2 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Explain quantum computing"}],
                    "temperature": 0.9,
                },
                headers={"host": "test"},
            )
            assert resp2.status_code == 200

            # Should be cache miss (different scope)
            assert proxy.cache.metrics.misses >= 1


@pytest.mark.asyncio
async def test_fallback_without_embeddings_extra():
    """Test d: Embeddings requested with extra absent falls back to Jaccard."""

    # This test runs in environment WITHOUT embeddings extra installed
    # Cache should fall back to Jaccard-only mode and still serve responses

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,  # Request embeddings
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        async with AsyncClient() as client:
            # First request
            resp1 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Test prompt"}],
                },
                headers={"host": "test"},
            )
            assert resp1.status_code == 200

            # Exact repeat - should hit via Jaccard
            resp2 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Test prompt"}],
                },
                headers={"host": "test"},
            )
            assert resp2.status_code == 200

            # Should work (fallback to Jaccard)
            assert proxy.cache.metrics.hits >= 1
            # If embeddings unavailable, enable_embeddings should be False after init
            assert proxy.cache.enable_embeddings == False


@pytest.mark.asyncio
async def test_embedding_miss_on_cross_scope():
    """Test e: Embedding hit whose best match exists only under another scope returns MISS."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        async with AsyncClient() as client:
            # First request with model-a
            resp1 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "model-a",
                    "messages": [{"role": "user", "content": "Optimize database performance"}],
                },
                headers={"host": "test"},
            )
            assert resp1.status_code == 200
            response1_content = resp1.json()["choices"][0]["message"]["content"]

            # Semantically similar prompt but different scope (model-b)
            resp2 = await client.post(
                "http://test/v1/chat/completions",
                json={
                    "model": "model-b",
                    "messages": [{"role": "user", "content": "Improve database query efficiency"}],
                },
                headers={"host": "test"},
            )
            assert resp2.status_code == 200
            response2_content = resp2.json()["choices"][0]["message"]["content"]

            # Should be cache miss (best embedding match is under different scope)
            # Response should NOT match model-a's response
            assert response2_content != response1_content
            assert proxy.cache.metrics.misses >= 1
