# Findings

## Phase 1: Pre-existing test failure (not caused by scope-aware caching fix)

Test: `tests/test_cli.py::ExperimentalTuningCliTests::test_collector_failures_keep_only_valid_complete_progress`

Failure:
```
AssertionError: 'failed' != 'complete'
- failed
+ complete
```

This failure exists before the ITEM 3 commit (f5a5842). It is unrelated to the scope-aware caching changes.

## Phase 2: Missing Backend Error Handling

**Finding:** proxy.py has no raise_for_status() or response validation logic.

**Question:** Can a backend error response containing a choices array be cached and served as valid? For example:
- Backend returns HTTP 500 but includes a choices array in JSON body
- Backend returns HTTP 200 with choices[0].finish_reason = "error"
- Backend returns HTTP 429 (rate limit) with partial response

**Current behavior:** All responses from backend are cached regardless of HTTP status code or content. The proxy trusts backend responses unconditionally.

**Test coverage:** test_backend_timeout_does_not_hang_waiters is skipped and documents that error paths are not tested.

**Not yet investigated.** Requires determining:
1. Which HTTP error codes should prevent caching
2. Whether finish_reason values indicate error states
3. Whether partial/incomplete responses should be cached

## Phase 2: Missing Backend Timeout Handling

**Finding:** proxy.py has no backend_timeout_seconds parameter or timeout configuration.

**Problem:** A hung backend hangs all waiters on the shared asyncio.Future. From proxy.py:342-441:

```python
async with self._inflight_lock:
    if dedup_key in self._inflight:
        # Request is already in-flight, wait for it
        future = self._inflight[dedup_key]
        is_waiter = True
```

If the primary request to the backend hangs, all concurrent waiters for that same prompt+scope hang indefinitely.

**Test coverage:** test_backend_timeout_does_not_hang_waiters (test_proxy_dedup.py:337) is skipped with reason "Timeout test requires full HTTP server fixture". Test documents this gap but does not enforce timeout behavior.

**httpx.AsyncClient:** Currently constructed with timeout=120.0 (proxy.py:77). This provides basic protection but:
1. Not configurable per-request or per-backend
2. No separate read vs connect timeout
3. No timeout for in-flight Future await

**Not yet investigated.** Requires determining:
1. Whether timeout should cancel Future and propagate to all waiters
2. Whether timed-out requests should retry or fail immediately
3. Whether timeout configuration belongs in ProxyServer.__init__ or per-request

## Phase 2: Async Transport Cleanup Warnings (RESOLVED)

**Finding:** Tests using proxy_to_mock fixture raise unraisable exception warnings in CI with PYTHONWARNINGS=error.

**Root cause:** Three tests in test_proxy_dedup.py created httpx.AsyncClient to fetch /health endpoint without using async context manager or calling .aclose(). pytest's gc_collect_harder during test session teardown surfaces these as ResourceWarning for unclosed transports.

**Resolution (commit 3fb3464):** Wrapped all three unclosed client instantiations with async context manager:
- test_five_concurrent_identical_requests_one_backend_call (line 184)
- test_ten_concurrent_paraphrases_current_behavior (line 250)
- test_concurrent_requests_with_mixed_prompts (line 461)

**Production verification:** SIGTERM test with 20 sequential cache hits (19 cache hits, 1 backend call) completed cleanly with no transport warnings, confirming production code (src/throttle/proxy.py) properly closes httpx.AsyncClient via ProxyServer.shutdown().

**CI run 32669184879 (post-fix):** 346 passed, 3 skipped, 0 failed, 0 errors across all Python versions.
