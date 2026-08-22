"""Throttle Proxy: OpenAI-compatible caching proxy with in-flight similarity deduplication."""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .cache import SimilarityCache, EMBEDDINGS_AVAILABLE, _get_embedding


class ProxyServer:
    """OpenAI-compatible proxy with semantic caching and request deduplication."""

    def __init__(
        self,
        backend_url: str,
        *,
        enable_cache: bool = True,
        cache_ttl_seconds: float = 3600.0,
        cache_max_size: int = 1000,
        cache_similarity_threshold: float = 0.85,
        backend_timeout_seconds: float = 120.0,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.enable_cache = enable_cache
        self.backend_timeout_seconds = backend_timeout_seconds
        self.cache: Optional[SimilarityCache] = None

        if self.enable_cache:
            self.cache = SimilarityCache(
                ttl_seconds=cache_ttl_seconds,
                max_size=cache_max_size,
                similarity_threshold=cache_similarity_threshold,
            )

        self.app = FastAPI(title="Throttle Proxy", version="0.3.0")
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.get("/health")(self.health)

        self._client: Optional[httpx.AsyncClient] = None

        # In-flight request tracking for deduplication
        # Maps prompt -> Future[response]
        self._inflight: Dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()

        self._backend_calls = 0

    async def startup(self):
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(timeout=self.backend_timeout_seconds)

    async def shutdown(self):
        """Cleanup HTTP client."""
        if self._client:
            await self._client.aclose()

    async def health(self):
        """Health check endpoint."""
        return {
            "status": "ok",
            "cache_enabled": self.enable_cache,
            "cache_stats": {
                "hits": self.cache.metrics.hits if self.cache else 0,
                "misses": self.cache.metrics.misses if self.cache else 0,
                "evictions": self.cache.metrics.evictions if self.cache else 0,
                "backend_calls": self._backend_calls,
            } if self.cache else None,
        }

    def _extract_scope_key(self, request_body: Dict[str, Any]) -> str:
        excluded = {"messages", "stream"}
        scope_params = {
            k: v for k, v in request_body.items()
            if k not in excluded
        }
        return json.dumps(scope_params, sort_keys=True)

    def _extract_prompt(self, request_body: Dict[str, Any]) -> str:
        messages = request_body.get("messages", [])
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                role = msg.get("role", "user")
                content = msg["content"]
                prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)

    async def _fake_stream_response(
        self, cached_response: Dict[str, Any]
    ) -> AsyncIterator[str]:
        first_chunk = {
            "id": cached_response.get("id", "cached-response"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": cached_response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        await asyncio.sleep(0.001)

        content = cached_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i+chunk_size]
            content_chunk = {
                "id": cached_response.get("id", "cached-response"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": cached_response.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            await asyncio.sleep(0.001)

        final_chunk = {
            "id": cached_response.get("id", "cached-response"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": cached_response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": cached_response.get("choices", [{}])[0].get("finish_reason", "stop"),
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def _make_backend_request(
        self,
        request_body: Dict[str, Any],
        headers: Dict[str, str],
        prompt: str,
        is_streaming: bool,
    ) -> Dict[str, Any]:
        self._backend_calls += 1

        if is_streaming:
            accumulated_content = []
            response_metadata = {}

            async with self._client.stream(
                "POST",
                f"{self.backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)

                            if not response_metadata:
                                response_metadata["id"] = chunk.get("id", "")
                                response_metadata["model"] = chunk.get("model", "")
                                response_metadata["created"] = chunk.get("created", int(time.time()))

                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                if "content" in delta:
                                    accumulated_content.append(delta["content"])
                                if choice.get("finish_reason"):
                                    response_metadata["finish_reason"] = choice["finish_reason"]
                        except json.JSONDecodeError:
                            pass

            full_content = "".join(accumulated_content)
            return {
                "id": response_metadata.get("id", ""),
                "object": "chat.completion",
                "created": response_metadata.get("created", int(time.time())),
                "model": response_metadata.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                        "finish_reason": response_metadata.get("finish_reason", "stop"),
                    }
                ],
            }
        else:
            response = await self._client.post(
                f"{self.backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    async def chat_completions(self, request: Request):
        try:
            request_body = await request.json()

            scope_key = self._extract_scope_key(request_body)
            prompt = self._extract_prompt(request_body)
            is_streaming = request_body.get("stream", False)

            # Check cache first (fast path)
            if self.enable_cache and self.cache:
                result = self.cache.get_with_key(prompt)
                if result is not None:
                    canonical_key, scope_dict = result
                    if isinstance(scope_dict, dict) and scope_key in scope_dict:
                        cached_response = scope_dict[scope_key]["response"]
                        if is_streaming:
                            return StreamingResponse(
                                self._fake_stream_response(cached_response),
                                media_type="text/event-stream",
                            )
                        else:
                            return JSONResponse(cached_response)

            # Cache miss - In-flight deduplication matching
            is_waiter = False
            future = None
            dedup_key = f"{scope_key}||{prompt}"

            # Phase 1: Fast lexical scan (Jaccard) under lock
            async with self._inflight_lock:
                matched_future = None
                for inflight_key, fut in list(self._inflight.items()):
                    if "||" in inflight_key:
                        inf_scope, inf_prompt = inflight_key.split("||", 1)
                        if inf_scope == scope_key:
                            if inf_prompt == prompt:
                                matched_future = fut
                                break
                            elif self.enable_cache and self.cache:
                                sim = self.cache._jaccard_similarity(prompt, inf_prompt)
                                if sim >= self.cache.similarity_threshold:
                                    matched_future = fut
                                    break
                
                if matched_future is not None:
                    future = matched_future
                    is_waiter = True

            # Phase 2: Slow semantic scan (ONNX) OUTSIDE lock, then reacquire lock
            # Guard: Only perform semantic embedding matching for queries longer than 3 words.
            if not is_waiter and EMBEDDINGS_AVAILABLE and len(prompt.split()) > 3:
                emb_q = _get_embedding(prompt)
                if emb_q is not None:
                    async with self._inflight_lock:
                        matched_future = None
                        for inflight_key, fut in list(self._inflight.items()):
                            if "||" in inflight_key:
                                inf_scope, inf_prompt = inflight_key.split("||", 1)
                                if inf_scope == scope_key and len(inf_prompt.split()) > 3:
                                    if inf_prompt == prompt:
                                        matched_future = fut
                                        break
                                    emb_inf = _get_embedding(inf_prompt)
                                    if emb_inf is not None:
                                        sim = float(np.dot(emb_q, emb_inf))
                                        threshold = self.cache.similarity_threshold if self.cache else 0.85
                                        if sim >= threshold:
                                            matched_future = fut
                                            break
                        
                        if matched_future is not None:
                            future = matched_future
                            is_waiter = True
                        else:
                            future = asyncio.Future()
                            self._inflight[dedup_key] = future
                            is_waiter = False

            # Phase 3: Final fallback for exact match registration
            if future is None:
                async with self._inflight_lock:
                    if dedup_key in self._inflight:
                        future = self._inflight[dedup_key]
                        is_waiter = True
                    else:
                        future = asyncio.Future()
                        self._inflight[dedup_key] = future
                        is_waiter = False

            # Execute request if primary, register result to Future
            if not is_waiter:
                headers = {
                    "Content-Type": "application/json",
                }
                if "authorization" in request.headers:
                    headers["Authorization"] = request.headers["authorization"]

                try:
                    response_data = await self._make_backend_request(
                        request_body, headers, prompt, is_streaming
                    )

                    if self.enable_cache and self.cache:
                        choices = response_data.get("choices", [])
                        if choices:
                            existing_scope_dict = self.cache.get_exact_no_metrics(prompt)
                            if existing_scope_dict is not None and isinstance(existing_scope_dict, dict):
                                scope_dict = existing_scope_dict
                            else:
                                scope_dict = {}

                            scope_dict[scope_key] = {
                                "_scope": scope_key,
                                "response": response_data,
                            }
                            self.cache.put(prompt, scope_dict)

                    future.set_result(response_data)

                except Exception as e:
                    future.set_exception(e)
                    raise
                finally:
                    async with self._inflight_lock:
                        self._inflight.pop(dedup_key, None)

            # Waiter resolves here
            response_data = await future

            if is_streaming:
                return StreamingResponse(
                    self._fake_stream_response(response_data),
                    media_type="text/event-stream",
                )
            else:
                return JSONResponse(response_data)

        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.json()
            except Exception:
                error_body = {"error": e.response.text or str(e)}
            return JSONResponse(content=error_body, status_code=e.response.status_code)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as e:
            return JSONResponse(content={"error": f"Backend timeout: {str(e)}"}, status_code=504)


def create_app(
    backend_url: str,
    enable_cache: bool = True,
    cache_ttl_seconds: float = 3600.0,
    cache_max_size: int = 1000,
    cache_similarity_threshold: float = 0.85,
    backend_timeout_seconds: float = 120.0,
) -> FastAPI:
    proxy = ProxyServer(
        backend_url,
        enable_cache=enable_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
        cache_similarity_threshold=cache_similarity_threshold,
        backend_timeout_seconds=backend_timeout_seconds,
    )

    @proxy.app.on_event("startup")
    async def startup_event():
        await proxy.startup()

    @proxy.app.on_event("shutdown")
    async def shutdown_event():
        await proxy.shutdown()

    return proxy.app
