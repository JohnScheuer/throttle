"""Test cache isolation when same model name served by different backends."""

import unittest
from unittest.mock import AsyncMock, MagicMock
import asyncio
import json

from fastapi import Request
from fastapi.testclient import TestClient
from throttle.proxy import ProxyServer


class TestBackendCacheIsolation(unittest.IsolatedAsyncioTestCase):
    """Test that same model name on different backends maintains cache isolation."""

    async def test_same_model_different_backends_isolated_responses(self):
        """Same model name on different backends should not share cache entries.

        This test verifies the fix for cache collision when:
        - Two backends both serve a model with the same name
        - Same prompt sent to each backend via model_backends routing
        - Each caller receives its own backend's response, not the other's

        Without the fix: Caller B receives backend A's cached response.
        With the fix: Each backend's responses are isolated by backend URL in scope_key.
        """
        # Setup: Two backends serving model "shared-model" with different responses
        model_backends = {
            "model-a": "http://backend-1:8000",
            "model-b": "http://backend-2:9000",
        }

        proxy = ProxyServer(
            backend_url="http://default:7000",
            enable_cache=True,
            cache_ttl_seconds=3600,
            cache_max_size=100,
            model_backends=model_backends,
        )

        # Mock responses from different backends with distinguishable content
        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = {
            "id": "resp-1",
            "model": "model-a",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "BACKEND_A_CONTENT"
                },
                "finish_reason": "stop"
            }],
        }

        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = {
            "id": "resp-2",
            "model": "model-b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "BACKEND_B_CONTENT"
                },
                "finish_reason": "stop"
            }],
        }

        async def mock_post(url, **kwargs):
            """Return different responses based on backend URL."""
            if "backend-1:8000" in url:
                return mock_response_1
            elif "backend-2:9000" in url:
                return mock_response_2
            else:
                raise AssertionError(f"Unexpected backend URL: {url}")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post)
        await proxy.startup()
        proxy._client = mock_client

        # Test: Send same prompt to both backends via chat_completions
        from fastapi import Request as FastAPIRequest
        from starlette.datastructures import Headers

        # Request A: to backend-1
        async def make_request_a():
            request_data_a = {
                "model": "model-a",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "stream": False,
            }

            # Create a mock Request object
            class MockRequest:
                def __init__(self, body):
                    self.body_data = body
                    self.headers = Headers({})

                async def json(self):
                    return self.body_data

            mock_request_a = MockRequest(request_data_a)
            return await proxy.chat_completions(mock_request_a)

        # Request B: to backend-2 (same prompt, different model name pointing to different backend)
        async def make_request_b():
            request_data_b = {
                "model": "model-b",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "stream": False,
            }

            class MockRequest:
                def __init__(self, body):
                    self.body_data = body
                    self.headers = Headers({})

                async def json(self):
                    return self.body_data

            mock_request_b = MockRequest(request_data_b)
            return await proxy.chat_completions(mock_request_b)

        # Make requests
        response_a = await make_request_a()
        response_b = await make_request_b()

        # Parse JSON responses
        response_a_data = json.loads(response_a.body.decode())
        response_b_data = json.loads(response_b.body.decode())

        # Assert: Caller A receives backend A's content
        self.assertEqual(
            response_a_data["choices"][0]["message"]["content"],
            "BACKEND_A_CONTENT",
            "Caller A should receive backend A's response"
        )

        # Assert: Caller B receives backend B's content (NOT backend A's cached response)
        self.assertEqual(
            response_b_data["choices"][0]["message"]["content"],
            "BACKEND_B_CONTENT",
            "Caller B should receive backend B's response, not backend A's cached response"
        )

        # Verify backend call count: 2 calls (one to each backend)
        calls = mock_client.post.call_args_list
        self.assertEqual(len(calls), 2, "Should make exactly 2 backend calls")
        self.assertIn("backend-1:8000", calls[0][0][0])
        self.assertIn("backend-2:9000", calls[1][0][0])

        await proxy.shutdown()

    async def test_scope_key_includes_backend_url(self):
        """Scope keys include backend URL when model_backends is non-empty."""
        model_backends = {
            "model-a": "http://backend-1:8000",
            "model-b": "http://backend-2:9000",
        }

        proxy = ProxyServer(
            backend_url="http://default:7000",
            enable_cache=True,
            model_backends=model_backends,
        )

        request_a = {
            "model": "model-a",
            "messages": [{"role": "user", "content": "test"}],
        }

        request_b = {
            "model": "model-b",
            "messages": [{"role": "user", "content": "test"}],
        }

        scope_key_a = proxy._extract_scope_key(request_a)
        scope_key_b = proxy._extract_scope_key(request_b)

        # Scope keys should differ
        self.assertNotEqual(scope_key_a, scope_key_b,
                            "Scope keys should differ when backends differ")

        # Should include backend URLs
        self.assertIn("backend-1:8000", scope_key_a)
        self.assertIn("backend-2:9000", scope_key_b)


if __name__ == "__main__":
    unittest.main()
