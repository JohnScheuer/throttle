# vLLM Backend Verification

This document describes how to verify Throttle proxy compatibility with vLLM using the integration test suite.

## Quick Start with Automated Script

For a fully automated verification of vLLM, SGLang, and LMDeploy, use the GPU backend verification script:

```bash
./validation/gpu_backend_verification.sh vllm
```

This script handles:
- vLLM installation
- Server launch with appropriate configuration
- Running all integration tests with correct environment variables
- Cleanup

For details on what the script does, continue reading the manual procedure below.

## Manual Verification Procedure

### Prerequisites

- Linux machine with NVIDIA GPU
- CUDA drivers installed
- Python 3.11 or later
- At least 8GB GPU memory for small models

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Launch vLLM Server

Use a small, fast model for testing. Recommended: `Qwen/Qwen2.5-0.5B-Instruct` (500M parameters)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8100 \
    --dtype half \
    --max-model-len 512
```

**Server startup time:** 30-90 seconds depending on hardware. Wait for the message:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8100
```

**Health check:**
```bash
curl http://localhost:8100/health
# Should return: {"status":"ok"}
```

### 3. Set Environment Variables

```bash
export BACKEND_URL="http://localhost:8100"
export BACKEND_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

These environment variables configure the integration tests to use vLLM instead of the default Ollama backend.

### 4. Run Integration Tests

**All 9 integration tests:**
```bash
python3 -m pytest tests/test_proxy_integration.py tests/test_proxy.py -v --tb=short
```

**Individual test files:**
```bash
# 7 tests from test_proxy_integration.py
python3 -m pytest tests/test_proxy_integration.py -v

# 2 tests from test_proxy.py
python3 -m pytest tests/test_proxy.py -v
```

### 5. What Constitutes PASS

**Test suite PASS criteria:**
- **Collected:** 9 tests
- **Passed:** 8 tests (88%)
- **Failed:** 1 test (11%)
- **Expected failure:** `test_backend_error_propagation`

**Expected output summary:**
```
====== 1 failed, 8 passed in XXs ======
```

**Why test_backend_error_propagation fails:** This test expects HTTP 404 errors to propagate through the proxy, but the proxy currently lacks `raise_for_status()` validation. This is a known gap documented in FINDINGS.md Phase 2 entry 1. The test failure confirms the proxy needs error handling improvements, but does NOT indicate vLLM incompatibility.

**The 8 passing tests verify:**
1. ✅ Cache miss followed by cache hit on exact repeat
2. ✅ Exact repeat hits, paraphrase misses (current similarity threshold)
3. ✅ Scope isolation with different parameters (model, temperature, max_tokens)
4. ✅ Cross-model cache isolation (different models maintain separate caches)
5. ✅ In-flight request deduplication (concurrent identical requests)
6. ✅ Streaming response caching
7. ✅ Cache hit with real HTTP client (end-to-end external client test)
8. ✅ Streaming with cache (fake-streaming from cached response)

### 6. Performance Measurements Beyond Pass/Fail

The integration test suite verifies correctness but does NOT capture performance metrics. To measure vLLM-specific performance:

#### A. Backend Latency (Time-to-First-Token on Cache Miss)

**Not currently captured by integration tests.** To measure:

1. Use the `throttle smoke` CLI with `--enable-cache`:
   ```bash
   export OPENAI_API_KEY=dummy
   throttle smoke \
       --model "Qwen/Qwen2.5-0.5B-Instruct" \
       --url http://localhost:8100 \
       --enable-cache \
       --max-tokens 10 \
       --stream \
       --allow-unknown-cost
   ```

2. Look for the `ttft_ms.mean` field in the JSON output (`throttle-report.json`):
   ```bash
   cat throttle-report.json | python3 -c "import json, sys; d=json.load(sys.stdin); print('Backend TTFT mean:', d['conditions'][0]['blocks'][0]['diagnostic_metrics']['ttft_ms']['mean'], 'ms')"
   ```

   **Example output:**
   ```
   Backend TTFT mean: 144.41 ms
   ```

   **Backend latency** = `ttft_ms.mean` value from the first condition (cache miss, concurrency 1)

#### B. Cache Lookup Cost

**Not directly reported in integration tests.** To estimate:

1. Run `throttle smoke` a second time (cache hits):
   ```bash
   export OPENAI_API_KEY=dummy
   throttle smoke \
       --model "Qwen/Qwen2.5-0.5B-Instruct" \
       --url http://localhost:8100 \
       --enable-cache \
       --max-tokens 10 \
       --stream \
       --allow-unknown-cost
   ```

2. Check the cached condition (concurrency 4 or 8):
   ```bash
   cat throttle-report.json | python3 -c "import json, sys; d=json.load(sys.stdin); print('Cache hit e2e p95:', d['conditions'][1]['blocks'][0]['diagnostic_metrics']['e2e_latency_ms']['p95'], 'ms')"
   ```

   **Note:** Cache hits have `ttft_ms.count = 0` (no backend token generation). Use end-to-end latency as proxy for cache lookup cost.

   **Cache lookup cost** ≈ e2e latency p95 for cached requests (typically 1-10ms)

#### C. Break-Even Hit Rate

**Not currently captured by existing tooling.** The break-even hit rate is the minimum cache hit rate needed for the proxy to provide net latency savings.

**Formula (from project specification):**

```
break_even_hit_rate = lookup_cost / backend_latency
```

Where:
- **lookup_cost** = In-process cache lookup latency (from step B, typically 1-10ms)
- **backend_latency** = Full backend response time (from step A, TTFT mean)

**To calculate manually:**

1. Measure **backend_latency** from step A (TTFT mean for cache misses)
2. Estimate **lookup_cost** from step B (e2e p95 for cache hits)
3. Calculate: `break_even_hit_rate = lookup_cost / backend_latency`

**Example (from actual Ollama measurements):**
- Backend latency (TTFT mean) = 144.41ms
- Cache lookup cost (e2e p95 for hits) ≈ 5ms (estimated)
- Break-even = 5ms / 144.41ms = 0.0346 (3.5%)

**Interpretation:** If >3.5% of requests are cache hits, the proxy provides net latency savings. Below this threshold, cache lookup overhead outweighs savings from avoided backend calls.

**What would be needed to capture this automatically:**
- Add `--measure-breakeven` flag to `throttle bench` that:
  1. Runs N unique prompts (misses) to measure average backend TTFT
  2. Runs M repeated prompts (hits) to measure average cache hit TTFT
  3. Calculates and reports break-even hit rate
  4. Outputs JSON with `backend_ttft_ms`, `cache_hit_ttft_ms`, `cache_overhead_ms`, `breakeven_hit_rate`

### 7. Known vLLM-Specific Considerations

**Tests that require multi-model support:**
- `test_scope_isolation_different_parameters` uses `llama3.2:3b` as a second model
- `test_scope_variants_coexisting` uses `llama3.2:3b` as a second model

If your vLLM deployment only has one model loaded, these tests will fail with 404. Solutions:
1. Load a second model in vLLM (e.g., `Qwen/Qwen2.5-1.5B-Instruct`)
2. Set `BACKEND_MODEL_2` environment variable (requires test suite modification)
3. Accept that 2 tests will fail (not a vLLM compatibility issue)

**Response shape:** vLLM uses OpenAI-compatible response format. No changes needed.

**Endpoint paths:** vLLM implements `/v1/chat/completions`, `/health`, and `/v1/models`. Fully compatible with test suite.

## Troubleshooting

**Issue:** `Connection refused` when running tests
**Solution:** Ensure vLLM server is running and listening on port 8100. Check server logs for startup errors.

**Issue:** `Model not found` error
**Solution:** Verify the model name matches exactly: `Qwen/Qwen2.5-0.5B-Instruct`. Check vLLM server logs to confirm model loaded successfully.

**Issue:** CUDA out of memory
**Solution:** Use a smaller model (e.g., `Qwen/Qwen2.5-0.5B-Instruct` instead of larger variants) or reduce `--max-model-len`.

**Issue:** Tests timeout
**Solution:** vLLM first-time model download can be slow. Wait for model download to complete before running tests. Subsequent runs will use cached model.
