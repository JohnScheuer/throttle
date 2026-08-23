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

## Phase 2: Async Transport Cleanup Warnings

**Finding:** Tests using proxy_to_mock fixture raise unraisable exception warnings in CI with PYTHONWARNINGS=error.

**Error pattern:**
```
pytest.PytestUnraisableExceptionWarning: Exception ignored while calling deallocator <function _SelectorTransport.__del__ at ...>: None
pytest.PytestUnraisableExceptionWarning: Exception ignored while finalizing socket <socket.socket fd=15, family=2, type=1, proto=6, laddr=('127.0.0.1', ...), raddr=('127.0.0.1', 58091)>: None
```

**Affected tests:**
- test_ten_concurrent_paraphrases_current_behavior
- test_sequential_identical_requests_hit_cache
- test_cross_scope_hit_inflation_bug_regression (Python 3.11, 3.12)
- test_safety_validation.py::AdversarialValidationTests::test_result_tampering_and_uninitialized_result_fail_projection (Python 3.13, 3.14)

**Root cause:** Lifespan context manager shutdown (await proxy.shutdown()) races with fixture teardown (server.should_exit = True; await task). Uvicorn lifespan calls proxy.shutdown() which closes httpx.AsyncClient, but then fixture force-terminates server before async transports fully close.

**CI run 32661178349:** 3 failed, 343 passed, 3 skipped across all Python versions.

**Not yet investigated.** Requires determining:
1. Whether lifespan shutdown should wait for all transports to close
2. Whether fixture should await server shutdown instead of forcing exit
3. Whether httpx.AsyncClient.aclose() needs explicit transport drain
