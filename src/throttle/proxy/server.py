"""HTTP proxy server that forwards requests to Anthropic API with logging and caching."""
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from aiohttp import web


class ThrottleProxy:
    def __init__(
        self,
        port: int = 8080,
        log_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ):
        self.port = port
        self.log_dir = log_dir or Path.home() / ".throttle" / "logs"
        self.cache_dir = cache_dir or Path.home() / ".throttle" / "cache"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Anthropic pricing per million tokens (as of current rates)
        self.pricing = {
            "claude-3-5-sonnet-20241022": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
            },
            "claude-3-5-sonnet-20240620": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
            },
            "claude-3-opus-20240229": {
                "input": 15.00,
                "output": 75.00,
                "cache_write": 18.75,
                "cache_read": 1.50,
            },
            "claude-3-sonnet-20240229": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
            },
            "claude-3-haiku-20240307": {
                "input": 0.25,
                "output": 1.25,
                "cache_write": 0.30,
                "cache_read": 0.03,
            },
        }

    def _get_log_file(self) -> Path:
        """Get today's log file path."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"throttle-{today}.jsonl"

    def _log_request(self, log_entry: Dict[str, Any]) -> None:
        """Append a log entry to today's log file."""
        log_file = self._get_log_file()
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost in dollars for a request."""
        if model not in self.pricing:
            # Default to Sonnet pricing if model not recognized
            model = "claude-3-5-sonnet-20241022"
        
        prices = self.pricing[model]
        cost = 0.0
        cost += (usage.get("input_tokens", 0) / 1_000_000) * prices["input"]
        cost += (usage.get("output_tokens", 0) / 1_000_000) * prices["output"]
        cost += (usage.get("cache_creation_input_tokens", 0) / 1_000_000) * prices["cache_write"]
        cost += (usage.get("cache_read_input_tokens", 0) / 1_000_000) * prices["cache_read"]
        return cost

    def _hash_request(self, request_body: Dict[str, Any]) -> str:
        """Create a hash of the request body for cache lookups."""
        # Create a stable JSON representation for hashing
        canonical = json.dumps(request_body, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _has_tool_use(self, response_body: Dict[str, Any]) -> bool:
        """Check if response contains any tool use blocks."""
        content = response_body.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return True
        return False

    def _extract_file_paths(self, request_body: Dict[str, Any]) -> list[str]:
        """Extract file paths mentioned in the request for cache invalidation."""
        paths = []
        messages = request_body.get("messages", [])
        
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                # Simple heuristic: look for common file path patterns
                # This is basic but works for the MVP
                import re
                # Match absolute paths and relative paths with extensions
                path_pattern = r'(?:^|[\s\'"(])((?:[./~])?[\w\-./]+\.\w+)(?:$|[\s\'"):,])'
                found_paths = re.findall(path_pattern, content)
                paths.extend(found_paths)
        
        return paths

    def _check_files_unchanged(self, paths: list[str], cached_time: float) -> bool:
        """Check if all referenced files haven't been modified since cache time."""
        for path_str in paths:
            try:
                # Expand user and resolve relative paths
                path = Path(path_str).expanduser()
                if not path.is_absolute():
                    # Try relative to current working directory
                    path = (Path.cwd() / path).resolve()
                
                if path.exists() and path.is_file():
                    mtime = path.stat().st_mtime
                    if mtime > cached_time:
                        return False
            except (OSError, ValueError):
                # If we can't check the file, assume it changed
                return False
        
        return True

    def _get_cached_response(
        self, request_hash: str, request_body: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Try to retrieve a cached response if valid."""
        cache_file = self.cache_dir / f"{request_hash}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r") as f:
                cache_entry = json.load(f)
            
            cached_time = cache_entry["timestamp"]
            now = time.time()
            
            # Check if cache is within 1 day
            if now - cached_time > 86400:  # 24 hours
                return None
            
            # Extract file paths and check if they've changed
            file_paths = self._extract_file_paths(request_body)
            if file_paths and not self._check_files_unchanged(file_paths, cached_time):
                return None
            
            return cache_entry
            
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _save_to_cache(
        self, request_hash: str, request_body: Dict[str, Any], response_body: Dict[str, Any]
    ) -> None:
        """Save a response to cache."""
        cache_file = self.cache_dir / f"{request_hash}.json"
        cache_entry = {
            "timestamp": time.time(),
            "request": request_body,
            "response": response_body,
        }
        
        with open(cache_file, "w") as f:
            json.dump(cache_entry, f)

    async def handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming proxy requests."""
        start_time = time.time()
        
        # Read the request body
        try:
            request_body = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        
        # Extract model from request
        model = request_body.get("model", "unknown")
        request_hash = self._hash_request(request_body)
        
        # Check cache first
        cache_hit = False
        cached_entry = self._get_cached_response(request_hash, request_body)
        
        if cached_entry:
            response_body = cached_entry["response"]
            cache_hit = True
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate savings
            usage = response_body.get("usage", {})
            saved_cost = self._calculate_cost(model, usage)
            
            # Log the cache hit
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "cache_hit": True,
                "latency_ms": latency_ms,
                "usage": usage,
                "cost_usd": 0.0,
                "saved_usd": saved_cost,
            }
            self._log_request(log_entry)
            
            return web.json_response(response_body)
        
        # Forward to Anthropic API
        headers = {
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
            "x-api-key": self.api_key,
            "content-type": "application/json",
        }
        
        # Copy other headers if present
        for header in ["anthropic-beta"]:
            if header in request.headers:
                headers[header] = request.headers[header]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.anthropic.com{request.path}",
                    json=request_body,
                    headers=headers,
                    timeout=300.0,  # 5 minute timeout
                )
                
                latency_ms = (time.time() - start_time) * 1000
                response_body = response.json()
                
                # Extract usage information
                usage = response_body.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                
                # Calculate cost
                cost = self._calculate_cost(model, usage)
                
                # Log the request
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "cache_hit": False,
                    "latency_ms": latency_ms,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": cache_creation_tokens,
                        "cache_read_input_tokens": cache_read_tokens,
                    },
                    "cost_usd": cost,
                    "saved_usd": 0.0,
                }
                self._log_request(log_entry)
                
                # Cache the response if it doesn't contain tool use
                if not self._has_tool_use(response_body):
                    self._save_to_cache(request_hash, request_body, response_body)
                
                return web.Response(
                    status=response.status_code,
                    body=response.content,
                    headers={"content-type": "application/json"},
                )
                
        except httpx.HTTPError as e:
            return web.Response(status=500, text=f"Proxy error: {str(e)}")

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "proxy": "throttle"})

    def create_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application()
        app.router.add_post("/v1/messages", self.handle_request)
        app.router.add_get("/health", self.health_check)
        return app

    def run(self) -> None:
        """Run the proxy server."""
        app = self.create_app()
        print(f"🚀 Throttle proxy starting on http://localhost:{self.port}")
        print(f"📁 Logs: {self.log_dir}")
        print(f"💾 Cache: {self.cache_dir}")
        print(f"🔑 API key: {'configured' if self.api_key else 'NOT CONFIGURED'}")
        print(f"\nForwarding requests to https://api.anthropic.com")
        print(f"Press Ctrl+C to stop\n")
        
        web.run_app(app, host="127.0.0.1", port=self.port, print=None)


def main() -> None:
    """Entry point for the proxy server."""
    import sys
    
    port = int(os.getenv("THROTTLE_PORT", "8080"))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    
    proxy = ThrottleProxy(port=port)
    
    if not proxy.api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment", file=sys.stderr)
        print("Run 'throttle-setup' first to configure your API key", file=sys.stderr)
        sys.exit(1)
    
    try:
        proxy.run()
    except KeyboardInterrupt:
        print("\n\n👋 Throttle proxy stopped")


if __name__ == "__main__":
    main()
