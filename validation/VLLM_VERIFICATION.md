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

**Measured by contributor benchmarks** (cache miss path overhead):

The cache lookup cost varies with cache size due to Jaccard similarity computation over all cached entries:

- **10 entries:** Jaccard scan 0.016ms + ONNX embedding overhead = **5.3ms total miss path**
- **100 entries:** Jaccard scan 0.175ms + ONNX embedding overhead = **6.0ms total miss path** (unmeasured, interpolated)
- **1,000 entries:** Jaccard scan 1.573ms + ONNX embedding overhead = **6.8ms total miss path**

**Source:** Contributor benchmark data measuring Jaccard token-overlap computation time and combined miss path latency including ONNX embedding generation.

**Note:** Integration tests use default cache size of 100 entries (`--cache-max-size 100`). For cache size 100, use **6.0ms** as lookup cost estimate.

**Cache hit latency:** Cannot be measured by existing tooling.

**What `throttle smoke --enable-cache` CAN measure for cached requests:**
- Cache hit confirmation: `cache_hit_count` field (e.g., `cache_hit_count: 8`)
- Token counts: `cache_completion_tokens` (e.g., `80`) and `gpu_completion_tokens: 0`

**What `throttle smoke --enable-cache` CANNOT measure for cached requests:**

Raw `e2e_latency_ms` field from cached condition (concurrency 4 and 8 in smoke run):
```json
{
  "count": 0,
  "mean": null,
  "p50": null,
  "p90": null,
  "p95": null,
  "p95_ci": {
    "analysis_sample_n": 0,
    "confidence": 0.95,
    "high": null,
    "low": null,
    "method": "bounded_percentile_bootstrap_requests",
    "n": 0,
    "resamples": 300
  },
  "p99": null
}
```

The harness confirms cache hits occurred but does not instrument cached response path for timing. Both `ttft_ms` and `e2e_latency_ms` show `count: 0` with all percentile fields null.

**Conclusion:** No existing tooling can measure end-to-end latency on cache hits through the proxy.

#### C. Break-Even Hit Rate

**Not currently captured by existing tooling.** The break-even hit rate is the minimum cache hit rate needed for the proxy to provide net latency savings.

**Formula (from project specification):**

```
break_even_hit_rate = lookup_cost / backend_latency
```

Where:
- **lookup_cost** = In-process cache lookup latency (from step B, 5.3ms to 6.8ms depending on cache size)
- **backend_latency** = Full backend response time (from step A, TTFT mean)

**To calculate manually:**

1. Measure **backend_latency** from step A (TTFT mean for cache misses)
2. Use **lookup_cost** from step B (measured miss path overhead including Jaccard scan and ONNX embedding)
3. Calculate: `break_even_hit_rate = lookup_cost / backend_latency`

**Example (from actual Ollama measurements with cache size 100):**
- Backend latency (TTFT mean) = 144.41ms
- Cache lookup cost (100 entries) = 6.0ms
- Break-even = 6.0ms / 144.41ms = 0.0416 (4.2%)

**Interpretation:** If >4.2% of requests are cache hits, the proxy provides net latency savings. Below this threshold, cache lookup overhead outweighs savings from avoided backend calls.

**Cache size impact:**
- Smaller cache (10 entries): 5.3ms / 144.41ms = 3.7% break-even
- Larger cache (1,000 entries): 6.8ms / 144.41ms = 4.7% break-even

**What would be needed to capture this automatically:**
- Add `--measure-breakeven` flag to `throttle bench` that:
  1. Runs N unique prompts (misses) to measure average backend TTFT
  2. Runs M repeated prompts (hits) to measure average cache hit TTFT
  3. Calculates and reports break-even hit rate
  4. Outputs JSON with `backend_ttft_ms`, `cache_hit_ttft_ms`, `cache_overhead_ms`, `breakeven_hit_rate`

### 7. Known vLLM-Specific Considerations

**Multi-model testing:**

vLLM limitation: Each vLLM server instance serves exactly one model. The test suite includes two tests that verify cross-model cache isolation:
- `test_scope_isolation_different_parameters`
- `test_scope_variants_coexisting`

These tests require two different models. Since Ollama supports multiple models on a single server but vLLM does not, the test suite now supports `BACKEND_URL_2` to route the second model to a separate server.

**For vLLM multi-model testing:**
1. Launch two vLLM servers on different ports (e.g., 8100 and 8101) with different models
2. Set `BACKEND_URL="http://localhost:8100"` and `BACKEND_URL_2="http://localhost:8101"`
3. Set `BACKEND_MODEL` to the first model name and `BACKEND_MODEL_2` to the second model name

The automated script `validation/gpu_backend_verification.sh vllm` handles this setup automatically.

**For Ollama (default behavior):**
- `BACKEND_URL_2` defaults to `BACKEND_URL` if not set
- Both models are served from the same Ollama server

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
