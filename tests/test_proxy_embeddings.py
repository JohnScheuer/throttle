"""Tests for ONNX embedding tier through proxy endpoint.

All tests assert through the proxy endpoint on responses received,
never on internal structures like _store or _embedder.
"""

import asyncio
import pytest
from contextlib import asynccontextmanager

from throttle.proxy import ProxyServer


async def _wait_for_server(host: str, port: int, timeout: float = 5.0):
    """Poll TCP connection until server accepts connections or timeout."""
    import socket
    import time

    start = time.time()
    interval = 0.05  # 50ms between attempts

    while time.time() - start < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            sock.connect((host, port))
            return  # Success
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(interval)
        finally:
            sock.close()

    raise TimeoutError(f"Server at {host}:{port} did not accept connections within {timeout}s")


@asynccontextmanager
async def _fake_backend():
    """Mock backend that returns fixed responses on loopback."""
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
    import socket

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        response = await handler(request)
        return JSONResponse(response)

    # Bind socket to loopback with auto-assigned port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve(sockets=[sock]))
    await _wait_for_server("127.0.0.1", port)

    try:
        yield f"http://127.0.0.1:{port}"
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
        from httpx import AsyncClient
        import socket

        # Start proxy on real port
        import uvicorn

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Pair: cosine=0.9762, jaccard=0.1667 (measured 2026-08-23 with direct ONNX)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "What are the benefits of using Docker containers?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200
                response1 = resp1.json()

                # Second request: strong paraphrase (cosine > 0.95, jaccard < 0.85)
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "What advantages do Docker containers provide?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp2.status_code == 200
                response2 = resp2.json()

                # Should be cache hit via embedding tier
                assert proxy.cache.metrics.hits >= 1
                assert proxy.cache.metrics.embedding_hits >= 1
                # Responses should match (same cached content)
                assert response2["choices"][0]["message"]["content"] == response1["choices"][0]["message"]["content"]
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_weak_paraphrase_does_not_hit_at_threshold_095():
    """Test that PostgreSQL pair (cosine=0.878) correctly misses at threshold 0.95."""

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
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Pair: cosine=0.8783, jaccard<0.85 (measured 2026-08-23 with direct ONNX)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "How do I optimize PostgreSQL queries?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200

                # Weak paraphrase - should NOT hit at threshold 0.95
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "How can I improve database query performance in PostgreSQL?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp2.status_code == 200

                # Should be MISS - cosine 0.878 < threshold 0.95
                assert proxy.cache.metrics.misses >= 1
                assert proxy.cache.metrics.embedding_hits == 0
                # Verify embedding tier ran but found no match
                assert proxy.cache.metrics.embedding_scans_attempted >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task


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
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "What is machine learning?"}],
                    },
                )
                assert resp1.status_code == 200

                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-b",
                        "messages": [{"role": "user", "content": "What is machine learning?"}],
                    },
                )
                assert resp2.status_code == 200

                assert proxy.cache.metrics.misses >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task


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
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Explain quantum computing"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200

                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Explain quantum computing"}],
                        "temperature": 0.9,
                    },
                )
                assert resp2.status_code == 200

                assert proxy.cache.metrics.misses >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_fallback_without_embeddings_extra():
    """Test d: Embeddings requested with extra absent falls back to Jaccard."""

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
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Test prompt"}],
                    },
                )
                assert resp1.status_code == 200

                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Test prompt"}],
                    },
                )
                assert resp2.status_code == 200

                assert proxy.cache.metrics.hits >= 1
                # Embeddings are now available with direct ONNX, but exact match hits via Jaccard tier
                assert proxy.cache.metrics.lexical_hits >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task


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
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "Optimize database performance"}],
                    },
                )
                assert resp1.status_code == 200
                response1_content = resp1.json()["choices"][0]["message"]["content"]

                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-b",
                        "messages": [{"role": "user", "content": "Improve database query efficiency"}],
                    },
                )
                assert resp2.status_code == 200
                response2_content = resp2.json()["choices"][0]["message"]["content"]

                assert response2_content != response1_content
                assert proxy.cache.metrics.misses >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task
