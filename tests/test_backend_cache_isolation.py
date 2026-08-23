"""Test cache isolation when same model name served by different backends."""

import unittest
from unittest.mock import AsyncMock, MagicMock
import asyncio

from throttle.proxy import ProxyServer


class TestBackendCacheIsolation(unittest.IsolatedAsyncioTestCase):
    """Test that same model name on different backends maintains cache isolation."""

    async def test_same_model_different_backends_isolated_responses(self):
        """Same model name on different backends should not share cache entries.

        This test verifies the fix for cache collision when:
        - Two backends both serve a model with the same name
        - Same prompt sent to each backend via model_backends routing
        - Each should get its own cached response, not collide

        Without the fix: Second request gets first backend's cached response.
        With the fix: Each backend's responses are isolated by backend URL in scope_key.
        """
        # Setup: Two backends serving model "shared-model" with different responses
        model_backends = {
            "shared-model": "http://backend-1:8000",
            "shared-model-on-backend-2": "http://backend-2:9000",
        }

        proxy = ProxyServer(
            backend_url="http://default:7000",
            enable_cache=True,
            cache_ttl_seconds=3600,
            cache_max_size=100,
            model_backends=model_backends,
        )

        # Mock responses from different backends
        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = {
            "id": "backend-1-response",
            "model": "shared-model",
            "choices": [{"message": {"role": "assistant", "content": "Response from backend 1"}}],
        }

        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = {
            "id": "backend-2-response",
            "model": "shared-model",  # Same model name
            "choices": [{"message": {"role": "assistant", "content": "Response from backend 2"}}],
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
        proxy._client = mock_client

        # Test: Verify scope keys are different for same model on different backends
        request_1 = {
            "model": "shared-model",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        }

        request_2 = {
            "model": "shared-model-on-backend-2",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        }

        scope_key_1 = proxy._extract_scope_key(request_1)
        scope_key_2 = proxy._extract_scope_key(request_2)

        # With the fix, scope keys should be different (include backend URL)
        self.assertNotEqual(scope_key_1, scope_key_2,
                            "Scope keys should differ when backends differ")
        self.assertIn("backend-1:8000", scope_key_1,
                      "Scope key should include backend-1 URL")
        self.assertIn("backend-2:9000", scope_key_2,
                      "Scope key should include backend-2 URL")

        # Also verify cache isolation by checking backend call behavior
        # We'll simulate cache puts and gets to verify isolation
        prompt = "What is 2+2?"

        # Simulate backend-1 response cached
        scope_dict_1 = {scope_key_1: {"_scope": scope_key_1, "response": mock_response_1.json()}}
        proxy.cache.put(prompt, scope_dict_1)

        # Try to get with backend-2's scope key - should miss
        result_2 = proxy.cache.get_with_key_no_metrics(prompt)
        if result_2:
            canonical_key, scope_dict = result_2
            # scope_key_2 should NOT be in the scope_dict
            self.assertNotIn(scope_key_2, scope_dict,
                             "Backend-2 scope should not find backend-1's cached response")


if __name__ == "__main__":
    unittest.main()
