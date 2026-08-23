"""Test that scope key is byte-identical when model_backends is empty."""

import unittest
import json

from throttle.proxy import ProxyServer


class TestScopeKeyBackwardsCompatibility(unittest.TestCase):
    """Verify scope key unchanged when model_backends is empty/None."""

    def test_scope_key_identical_when_no_model_backends(self):
        """Scope key must be byte-identical to legacy behavior when model_backends empty."""
        # Create proxy with no model_backends (default behavior)
        proxy_no_mapping = ProxyServer(
            backend_url="http://localhost:8000",
            enable_cache=True,
        )

        # Create proxy with empty model_backends dict
        proxy_empty_mapping = ProxyServer(
            backend_url="http://localhost:8000",
            enable_cache=True,
            model_backends={},
        )

        request = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        }

        # Extract scope keys
        scope_key_no_mapping = proxy_no_mapping._extract_scope_key(request)
        scope_key_empty_mapping = proxy_empty_mapping._extract_scope_key(request)

        # Should be byte-identical
        self.assertEqual(scope_key_no_mapping, scope_key_empty_mapping)

        # Parse to verify structure
        scope_params = json.loads(scope_key_no_mapping)

        # Should NOT contain _backend_url
        self.assertNotIn("_backend_url", scope_params,
                         "Backend URL should not be in scope when model_backends empty")

        # Should contain model and other params
        self.assertEqual(scope_params["model"], "gpt-4")
        self.assertEqual(scope_params["temperature"], 0.7)
        self.assertEqual(scope_params["max_tokens"], 100)

        # Should NOT contain messages or stream (excluded)
        self.assertNotIn("messages", scope_params)
        self.assertNotIn("stream", scope_params)

    def test_scope_key_includes_backend_url_when_mapping_present(self):
        """Scope key includes backend URL when model_backends is non-empty."""
        proxy_with_mapping = ProxyServer(
            backend_url="http://localhost:8000",
            enable_cache=True,
            model_backends={"gpt-4": "http://backend-1:9000"},
        )

        request = {
            "model": "gpt-4",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "test"}],
        }

        scope_key = proxy_with_mapping._extract_scope_key(request)
        scope_params = json.loads(scope_key)

        # Should contain _backend_url
        self.assertIn("_backend_url", scope_params)
        self.assertEqual(scope_params["_backend_url"], "http://backend-1:9000")

    def test_legacy_scope_key_format(self):
        """Verify exact legacy scope key format for regression detection."""
        proxy = ProxyServer(
            backend_url="http://localhost:8000",
            enable_cache=True,
        )

        request = {
            "model": "llama3.2:1b",
            "temperature": 0.5,
            "messages": [{"role": "user", "content": "test"}],
        }

        scope_key = proxy._extract_scope_key(request)

        # Expected format: JSON with sorted keys, no _backend_url
        expected = json.dumps({"model": "llama3.2:1b", "temperature": 0.5}, sort_keys=True)

        self.assertEqual(scope_key, expected,
                         "Scope key format must match legacy behavior exactly")


if __name__ == "__main__":
    unittest.main()
