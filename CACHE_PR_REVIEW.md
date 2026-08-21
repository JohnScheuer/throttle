# Cache MVP PR #9 — Technical Review

**Reviewer:** Kush
**Date:** August 21, 2026
**PR:** #9 (feat: In-Memory Similarity Cache MVP)
**Author:** João (@JohnScheuer)
**Branch:** feat/cache-layer-mvp
**Status:** Draft (opened for feedback)

---

## OVERVIEW

João delivered a **functional cache MVP** with clean implementation:
- ✅ `SimilarityCache` class with Jaccard similarity matching
- ✅ Thread-safe with proper locking
- ✅ TTL + max-size FIFO eviction
- ✅ Unit tests (6 test cases)
- ✅ CLI flags (--enable-cache, --cache-ttl-seconds, --cache-max-size, --cache-similarity-threshold)
- ✅ Fully wired into _native_request with cache hit fast-path
- ✅ Both streaming and non-streaming paths populate cache

**Files changed:** 5 files, +260 lines, -28 deletions
- src/throttle/cache.py (90 lines, new)
- tests/test_cache.py (54 lines, new)
- src/throttle/benchmark.py (+105 changes)
- src/throttle/cli.py (+35 changes)
- src/throttle/models.py (+4 fields)

---

## STRENGTHS

### 1. Clean Cache Implementation ✅

**cache.py** (90 lines):
- Simple, focused API: `get()`, `put()`, `metrics`
- Jaccard similarity for fuzzy matching (0.85 default threshold)
- Thread-safe with `Lock()`
- Proper validation of constructor params
- TTL-based eviction + max-size FIFO eviction
- Explicit metrics tracking: `hits`, `misses`, `evictions`

**Code Quality:** Excellent. No dependencies, pure Python, easy to test.

### 2. Good Test Coverage ✅

**test_cache.py** (54 lines):
- 6 test cases covering:
  - Constructor validation
  - Exact match
  - Similarity match (Jaccard threshold)
  - Cache miss
  - TTL eviction
  - Max-size FIFO eviction
- All tests use realistic scenarios

**Missing tests (minor):**
- Thread safety (concurrent get/put)
- Edge cases (empty prompts, very long prompts)
- Metrics accuracy after evictions

**Verdict:** Good enough for MVP, can expand later.

### 3. Proper CLI Integration ✅

**cli.py changes:**
- 4 new flags: --enable-cache, --cache-ttl-seconds, --cache-max-size, --cache-similarity-threshold
- All flags have sensible defaults
- Properly integrated into _build_config()
- Help text is clear

**Models.py changes:**
- 4 new fields added to RunConfig (appended to preserve positional compatibility)

**Verdict:** Clean, backward-compatible.

### 4. Cache Hit Fast-Path is Correct ✅

**benchmark.py lines 720-746:**
```python
if cache_instance:
    # Extract user prompt from messages
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    if user_prompt:
        cached_data = cache_instance.get(user_prompt)
        if cached_data is not None:
            # CACHE HIT! Return mock RequestResult
            latency = time.perf_counter() - started
            comp_tokens = cached_data.get("completion_tokens", 10)
            return RequestResult(
                200,
                latency,
                completion_tokens=comp_tokens,
                prompt_tokens=cached_data.get("prompt_tokens", len(user_prompt.split())),
                finish_reason="stop",
                ttft_seconds=latency * 0.9 if config.stream else None,
                tpot_seconds=0.0001 if config.stream and comp_tokens > 1 else None,
                inter_chunk_seconds=tuple([0.0001] * max(0, comp_tokens - 1)) if config.stream else (),
                response_bytes=cached_data.get("response_bytes", 100),
            )
```

**What João is asking about:** Is this mock RequestResult approach acceptable?

**Analysis:**
- ✅ Cache hit latency is **real** (time.perf_counter() - started) — includes cache lookup overhead
- ✅ Token counts are **preserved** from original response (completion_tokens, prompt_tokens, response_bytes)
- ✅ finish_reason is hardcoded to "stop" (reasonable for cache hits)
- ⚠️ **TTFT is synthetic** (latency * 0.9) — not real TTFT from GPU
- ⚠️ **TPOT is synthetic** (0.0001 seconds = 0.1ms) — not real TPOT from GPU
- ⚠️ **inter_chunk_seconds is synthetic** (0.0001 per token) — not real chunk timing

---

## CRITICAL ISSUE: CACHE HITS WILL SKEW LATENCY PERCENTILES ⚠️

### The Problem

Cache hits return **near-instant latency** (~1-10ms for cache lookup) compared to **real GPU latency** (50-500ms+ for inference).

**Impact on metrics:**
- **TTFT p95:** Will be artificially LOW if many cache hits (p95 might be 5ms instead of 200ms)
- **TPOT p95:** Will be artificially LOW (0.1ms instead of 10-50ms)
- **E2E latency p95:** Will be artificially LOW
- **Throughput:** Will be artificially HIGH (requests complete instantly from cache)

**Why this matters for Throttle:**
- **Golden protocol** uses p95 latency for decision eligibility
- **SLO validation** compares p95 against thresholds
- **Comparison** computes deltas between baseline/candidate
- **Cached runs vs. uncached runs will not be comparable**

### Example Scenario

```
Run A (cache disabled):
- 200 requests, all to GPU
- E2E p95: 250ms
- TTFT p95: 120ms
- Throughput: 80 tokens/sec

Run B (cache enabled, 60% hit rate):
- 200 requests, 120 from cache, 80 to GPU
- E2E p95: 8ms (dominated by cache hits!)
- TTFT p95: 5ms (cache lookups)
- Throughput: 300 tokens/sec (cached requests are instant)

Golden comparison: Run B vs. Run A = "+275% throughput, -97% latency"
Reality: This is NOT a valid GPU optimization comparison!
```

---

## RECOMMENDED SOLUTIONS

### Option 1: EXCLUDE Cache Hits from Percentile Calculations (RECOMMENDED)

**Approach:** Tag cache hits, exclude them from latency/TTFT/TPOT percentiles, but count them for throughput.

**Implementation:**
1. Add `cache_hit: bool = False` field to RequestResult dataclass (benchmark.py:591)
2. Set `cache_hit=True` in the mock RequestResult (line 735)
3. In `_condition_metrics()` (benchmark.py:1460-1640), filter results:
   ```python
   # Separate cache hits from GPU requests
   gpu_results = [r for r in results if not r.cache_hit]
   cache_hits = [r for r in results if r.cache_hit]

   # Compute latency percentiles ONLY on GPU requests
   if gpu_results:
       ttft_p95 = percentile([r.ttft_seconds for r in gpu_results if r.ttft_seconds], 0.95)
   else:
       ttft_p95 = None  # All cache hits!

   # Count ALL requests (cache + GPU) for throughput
   total_tokens = sum(r.completion_tokens for r in results if r.valid)
   throughput = total_tokens / wall_seconds
   ```

4. Add to run_totals:
   ```json
   "run_totals": {
       "elapsed_seconds": 123.45,
       "offered_requests": 201,
       "completed_requests": 201,
       "failed_requests": 0,
       "cache_enabled": true,
       "cache_hits": 120,
       "cache_misses": 81,
       "cache_hit_rate": 0.5970,
       "gpu_requests": 81,  // NEW: requests that actually hit GPU
       "cache_requests": 120  // NEW: requests served from cache
   }
   ```

**Pros:**
- ✅ Preserves GPU-only latency percentiles (decision-grade comparisons remain valid)
- ✅ Throughput accurately reflects cache benefit (total tokens / wall time)
- ✅ Clear separation of cache vs. GPU metrics
- ✅ Golden can validate that baseline/candidate have same cache hit rate

**Cons:**
- ⚠️ Percentiles become undefined if ALL requests are cache hits (need to handle gracefully)
- ⚠️ Adds complexity to metrics aggregation

**Verdict:** This is the **correct approach** for decision-grade benchmarking.

---

### Option 2: Separate Cache Metrics Section (Alternative)

**Approach:** Report cache hits separately, keep all latencies in main metrics.

**Implementation:**
Add new top-level section in report:
```json
"cache_metrics": {
    "enabled": true,
    "hits": 120,
    "misses": 81,
    "hit_rate": 0.5970,
    "cache_latency_ms": {
        "mean": 2.3,
        "p50": 1.8,
        "p95": 4.5,
        "p99": 7.2
    },
    "gpu_latency_ms": {
        "mean": 185.4,
        "p50": 180.2,
        "p95": 245.8,
        "p99": 312.1
    }
}
```

**Pros:**
- ✅ Explicit separation of cache vs. GPU
- ✅ Can compare cache effectiveness across runs
- ✅ Doesn't break existing metrics schema

**Cons:**
- ⚠️ Duplicates latency reporting
- ⚠️ Doesn't solve Golden comparison problem (need to validate cache_hit_rate match anyway)

**Verdict:** Good for observability, but still needs Option 1's filtering logic.

---

### Option 3: Disable Cache in Decision-Grade Modes (Temporary Fix)

**Approach:** Only allow --enable-cache in smoke/benchmark modes, reject in Golden.

**Implementation:**
```python
# In golden.py:golden_preflight_reasons()
if config.enable_cache:
    reasons.append("golden_requires_cache_disabled")
```

**Pros:**
- ✅ Quick fix to prevent invalid Golden runs
- ✅ Maintains decision integrity immediately

**Cons:**
- ❌ Defeats the purpose of cache (can't measure cache benefit in Golden)
- ❌ Limits cache to exploratory modes only

**Verdict:** Acceptable **short-term** until Option 1 is implemented, but not final solution.

---

## SPECIFIC FEEDBACK FOR JOÃO

### 1. Mock RequestResult Approach

**Question from João:**
> "Take a look at how I handled the mock RequestResult on a cache hit—I want to ensure it aligns with how you want the telemetry represented without skewing the underlying GPU percentiles."

**Answer:**
Your implementation is **functionally correct** for MVP, but the synthetic latencies (TTFT, TPOT, inter_chunk_seconds) **will skew GPU percentiles** and make cache-enabled runs **incomparable** with cache-disabled runs.

**Required changes:**
1. Add `cache_hit: bool = False` field to RequestResult (line 591-614)
2. Set `cache_hit=True` in your mock RequestResult (line 735)
3. Filter cache hits from latency percentile calculations (in _condition_metrics)
4. Add cache stats to run_totals

See **Option 1** above for detailed implementation.

---

### 2. Missing: Cache Stats in run_totals

**Current state:** Cache hits/misses are NOT reported in the final JSON.

**Why this matters:**
- Operators need to see cache effectiveness
- Golden validation needs to verify both runs have same hit rate
- Cost calculations need to distinguish cached vs. API requests

**Required addition:**
In `_finalize_report()` (around line 2213), add:
```python
run_totals = budget.public_dict()
run_totals["elapsed_seconds"] = final_elapsed

# ADD THIS:
if global_cache:
    run_totals["cache_enabled"] = True
    run_totals["cache_hits"] = global_cache.metrics.hits
    run_totals["cache_misses"] = global_cache.metrics.misses
    run_totals["cache_hit_rate"] = (
        global_cache.metrics.hits / (global_cache.metrics.hits + global_cache.metrics.misses)
        if (global_cache.metrics.hits + global_cache.metrics.misses) > 0
        else 0.0
    )
else:
    run_totals["cache_enabled"] = False
    run_totals["cache_hits"] = 0
    run_totals["cache_misses"] = 0
    run_totals["cache_hit_rate"] = 0.0

report["run_totals"] = run_totals
```

**Status:** ❌ Missing from current PR, needs to be added.

---

### 3. Debug Print Statement

**Line 732:**
```python
print(f"\n🚀 CACHE HIT! Bypassing API for prompt: {user_prompt[:30]}...") # PRINT TEMPORÁRIO DE TESTE
```

**Feedback:** Remove before merging. Use logging if needed for debugging.

**Suggestion:**
```python
# Option A: Remove entirely
# Option B: Use logging
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Cache hit for prompt: {user_prompt[:30]}...")
```

---

### 4. Cache Key Design: User Prompt Only

**Current approach (lines 723-726):**
```python
for msg in reversed(messages):
    if msg.get("role") == "user":
        user_prompt = msg.get("content", "")
        break
```

**Limitation:** Only caches based on **last user message**, ignoring conversation history.

**Example where this breaks:**
```
Messages A: [{"role": "system", "content": "You are a poet"}, {"role": "user", "content": "Write a haiku"}]
Messages B: [{"role": "system", "content": "You are a coder"}, {"role": "user", "content": "Write a haiku"}]

Current cache: Both get same cache hit (user prompt is identical)
Correct behavior: Different system prompts should NOT share cache
```

**Recommended fix (for next iteration):**
```python
# Hash the entire message sequence
import hashlib
import json

def _messages_cache_key(messages: Prompt) -> str:
    canonical = json.dumps(list(messages), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

# Use in cache lookup
cache_key = _messages_cache_key(messages)
cached_data = cache_instance.get(cache_key)
```

**Status:** ⚠️ **Acceptable for MVP** (most benchmarks have fixed system prompts), but document as **KNOWN_GAP** and fix before v0.3.0 release.

---

### 5. Golden Protocol Validation

**Missing validation:** Golden must ensure baseline/candidate have **same cache_enabled setting**.

**Where to add:** In `golden.py:_session_eligibility_reasons()` (around line 1040):
```python
# Check cache_enabled consistency
cache_enabled_values = {
    _path(report, "run_totals", "cache_enabled") for report in reports
}
if len(cache_enabled_values) > 1:
    reasons.append("cache_enabled_mismatch_across_positions")
```

**Also check:** Cache hit rates should be similar (within tolerance) across positions:
```python
if all(_path(r, "run_totals", "cache_enabled") for r in reports):
    hit_rates = [_path(r, "run_totals", "cache_hit_rate") for r in reports]
    if max(hit_rates) - min(hit_rates) > 0.10:  # 10% tolerance
        reasons.append("cache_hit_rate_variance_too_high")
```

**Status:** ❌ Missing, required for cache-enabled Golden runs.

---

## OVERALL ASSESSMENT

### Code Quality: **A-**
- Clean implementation
- Good test coverage
- Thread-safe
- Proper eviction policies

### Integration: **B**
- ✅ CLI flags work correctly
- ✅ Wired into request flow properly
- ❌ Missing cache stats in run_totals
- ❌ Missing cache_hit field in RequestResult
- ❌ No Golden validation for cache_enabled

### Telemetry Design: **C+**
- ⚠️ **CRITICAL:** Synthetic latencies will skew GPU percentiles
- ⚠️ Cache hits not tagged, mixed with GPU requests in metrics
- ⚠️ Cannot compare cache-enabled vs. cache-disabled runs validly
- ❌ No separation of cache vs. GPU latency distributions

---

## ACTION ITEMS FOR JOÃO

### Must-Fix (Blocking Merge)

1. **Add cache_hit field to RequestResult**
   - Location: benchmark.py:591
   - Add: `cache_hit: bool = False`
   - Set to True in mock RequestResult (line 735)

2. **Filter cache hits from latency percentiles**
   - Location: benchmark.py:1460-1640 (_condition_metrics)
   - Separate gpu_results from cache_hits
   - Compute TTFT/TPOT/E2E percentiles ONLY on GPU requests
   - Count both for throughput

3. **Add cache stats to run_totals**
   - Location: benchmark.py:~2213 (_finalize_report)
   - Add: cache_enabled, cache_hits, cache_misses, cache_hit_rate

4. **Remove debug print**
   - Location: benchmark.py:732
   - Delete or replace with logging

### Should-Fix (Before v0.3.0)

5. **Fix cache key to include full message context**
   - Current: Only last user message
   - Correct: Hash entire messages array
   - Impact: Prevents incorrect cache hits with different system prompts

6. **Add Golden cache validation**
   - Location: golden.py:~1040
   - Check: cache_enabled matches across all 6 positions
   - Check: cache_hit_rate variance within tolerance

7. **Document cache limitations in KNOWN_GAPS.md**
   - Limitation 16: "Cache key uses last user message only, ignoring conversation history"
   - Limitation 17: "Cache-enabled Golden runs require manual verification of similar hit rates"

### Nice-to-Have (Future)

8. **Add cache size/memory metrics**
   - Track bytes cached
   - Warn if cache size exceeds memory limits

9. **Add cache warming API**
   - Support --cache-policy warm with pre-population

10. **Thread-safety tests**
    - Concurrent get/put from multiple tasks

---

## SCHEMA COMPLIANCE CHECK

### Does this require schema version bump? ❌ NO

**Reason:** All cache-related fields are **additive**:
- `cache_enabled` defaults to false
- `cache_hits`, `cache_misses`, `cache_hit_rate` default to 0
- `cache_hit` field in RequestResult defaults to False
- Existing reports without these fields remain valid

**CACHE_INTEGRATION_GUIDE.md compliance:** ✅ Matches recommended approach

---

## MERGE RECOMMENDATION

**Status:** **DO NOT MERGE YET** — Needs critical fixes

**Severity of issues:**
- 🔴 **CRITICAL:** Latency percentiles will be incorrect (breaks decision-grade comparisons)
- 🟡 **HIGH:** Missing cache stats in run_totals (no visibility into cache effectiveness)
- 🟡 **HIGH:** Missing Golden validation (invalid cache-enabled Golden runs possible)
- 🟢 **LOW:** Debug print, cache key limitation (acceptable for MVP)

**Estimated effort to fix:**
- Items 1-4 (must-fix): **2-3 hours**
- Items 5-7 (should-fix): **3-4 hours**
- Total: **1 day of focused work**

**Timeline:**
- João fixes items 1-4: **Tomorrow (Aug 22)**
- Review & iterate: **Aug 23**
- Merge: **Aug 24**
- Items 5-7 in follow-up PR: **Aug 25-26**

---

## POSITIVE FEEDBACK

**What João did exceptionally well:**
1. ✅ **Fast delivery** — Pivoted from cache_viability to cache.py and shipped in <3 days
2. ✅ **Clean code** — cache.py is 90 lines, easy to read, no dependencies
3. ✅ **Good tests** — 6 test cases covering key scenarios
4. ✅ **Proper threading** — Used Lock() correctly for thread safety
5. ✅ **Correct integration** — Wired into both streaming and non-streaming paths
6. ✅ **Sensible defaults** — TTL=3600s, max_size=1000, threshold=0.85 are reasonable

**Overall:** This is **high-quality MVP code** that just needs telemetry fixes to be merge-ready.

---

## NEXT STEPS

1. **João:** Address must-fix items 1-4 (1 day)
2. **Kush:** Review updated PR (same day)
3. **João:** Merge to main (after approval)
4. **João:** Follow-up PR for items 5-7 (1-2 days)
5. **Both:** Update README to remove cache from OUT OF SCOPE
6. **Both:** Prep v0.3.0 release announcement

**Estimated time to merge:** **2-3 days** (optimistic)

---

**End of Review**

---

## APPENDIX: Example Updated Code

### RequestResult with cache_hit field

```python
@dataclass
class RequestResult:
    status_code: int | None
    e2e_seconds: float
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    ttft_seconds: float | None = None
    tpot_seconds: float | None = None
    inter_chunk_seconds: tuple[float, ...] = ()
    response_bytes: int = 0
    error_code: str | None = None
    cache_hit: bool = False  # NEW

    @property
    def valid(self) -> bool:
        return (
            self.status_code == 200
            and self.error_code is None
            and self.completion_tokens is not None
            and self.completion_tokens > 0
            and self.prompt_tokens is not None
            and self.finish_reason is not None
        )
```

### Mock RequestResult with cache_hit=True

```python
if cached_data is not None:
    # CACHE HIT!
    latency = time.perf_counter() - started
    comp_tokens = cached_data.get("completion_tokens", 10)
    return RequestResult(
        200,
        latency,
        completion_tokens=comp_tokens,
        prompt_tokens=cached_data.get("prompt_tokens", len(user_prompt.split())),
        finish_reason="stop",
        ttft_seconds=latency * 0.9 if config.stream else None,
        tpot_seconds=0.0001 if config.stream and comp_tokens > 1 else None,
        inter_chunk_seconds=tuple([0.0001] * max(0, comp_tokens - 1)) if config.stream else (),
        response_bytes=cached_data.get("response_bytes", 100),
        cache_hit=True,  # NEW: Tag this as a cache hit
    )
```

### Filtered metrics in _condition_metrics

```python
def _condition_metrics(...) -> dict[str, Any]:
    # Separate cache hits from GPU requests
    gpu_results = [r for r in results if r.valid and not r.cache_hit]
    cache_results = [r for r in results if r.valid and r.cache_hit]

    # Compute latency percentiles ONLY on GPU requests
    if gpu_results:
        ttft_values = [r.ttft_seconds for r in gpu_results if r.ttft_seconds is not None]
        ttft_p95 = percentile(ttft_values, 0.95) if ttft_values else None
    else:
        ttft_p95 = None  # All requests were cache hits

    # Throughput includes ALL valid requests (cache + GPU)
    all_valid = [r for r in results if r.valid]
    total_tokens = sum(r.completion_tokens for r in all_valid)
    throughput = total_tokens / wall_seconds if wall_seconds > 0 else 0.0

    return {
        "ttft_ms": {"p95": ttft_p95 * 1000 if ttft_p95 else None},
        "output_tokens_per_second": throughput,
        "cache_hits": len(cache_results),
        "gpu_requests": len(gpu_results),
        # ... rest of metrics
    }
```

### Cache stats in _finalize_report

```python
def _finalize_report(...):
    # ... existing code ...

    run_totals = budget.public_dict()
    run_totals["elapsed_seconds"] = final_elapsed

    # Add cache metrics
    if global_cache:
        total_cache_requests = global_cache.metrics.hits + global_cache.metrics.misses
        run_totals["cache_enabled"] = True
        run_totals["cache_hits"] = global_cache.metrics.hits
        run_totals["cache_misses"] = global_cache.metrics.misses
        run_totals["cache_hit_rate"] = (
            global_cache.metrics.hits / total_cache_requests
            if total_cache_requests > 0
            else 0.0
        )
    else:
        run_totals["cache_enabled"] = False
        run_totals["cache_hits"] = 0
        run_totals["cache_misses"] = 0
        run_totals["cache_hit_rate"] = 0.0

    report["run_totals"] = run_totals
    # ... rest of finalize ...
```
