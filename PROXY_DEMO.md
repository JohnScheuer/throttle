# Throttle Proxy - OpenAI-Compatible Caching Proxy

The Throttle proxy is a lightweight HTTP server that sits in front of real inference backends (vLLM, Ollama, SGLang, LMDeploy) and caches responses using semantic similarity matching.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for `/v1/chat/completions` endpoint
- **Semantic Caching**: Uses Jaccard similarity for prompt matching (not exact string matching)
- **Streaming Support**: Handles both streaming and non-streaming requests
- **Real-time Cache**: In-memory cache with configurable TTL and max size
- **Zero Backend Changes**: Works with any OpenAI-compatible backend

## Quick Start

### 1. Start the Proxy

```bash
throttle proxy \
  --backend-url http://localhost:11434 \
  --enable-cache \
  --port 8080 \
  --cache-ttl-seconds 3600 \
  --cache-max-size 1000 \
  --cache-similarity-threshold 0.85
```

This starts a proxy server on `http://localhost:8080` that forwards requests to Ollama running on `http://localhost:11434`.

### 2. Send Requests Through the Proxy

**Using curl (non-streaming):**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50,
    "stream": false
  }'
```

**Using the OpenAI Python client:**

```python
from openai import OpenAI

# Point the client to the proxy instead of the backend
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Proxy passes through to backend
)

# First request - cache miss, goes to backend
response1 = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=50
)
print(response1.choices[0].message.content)

# Second request - cache hit, instant response
response2 = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=50
)
print(response2.choices[0].message.content)
```

### 3. Check Cache Stats

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "cache_enabled": true,
  "cache_stats": {
    "hits": 15,
    "misses": 8,
    "evictions": 0
  }
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend-url` | *required* | Backend inference server URL |
| `--host` | `127.0.0.1` | Proxy server host |
| `--port` | `8080` | Proxy server port |
| `--enable-cache` | `false` | Enable semantic caching |
| `--cache-ttl-seconds` | `3600.0` | Cache entry TTL in seconds |
| `--cache-max-size` | `1000` | Maximum cache entries (FIFO eviction) |
| `--cache-similarity-threshold` | `0.85` | Jaccard similarity threshold (0.0-1.0) |

## How It Works

### Cache Miss Flow
1. Client sends request to proxy
2. Proxy checks cache using Jaccard similarity
3. **No match found** → Forward request to backend
4. Backend processes request and returns response
5. Proxy caches response and returns to client

### Cache Hit Flow
1. Client sends request to proxy
2. Proxy checks cache using Jaccard similarity
3. **Match found** (similarity ≥ threshold)
4. Proxy returns cached response immediately
5. No backend request needed

### Streaming Behavior
- **Cache miss**: Stream is passed through from backend in real-time
- **Cache hit**: Cached response is "fake-streamed" to maintain client compatibility

## Performance

Validated test results with Ollama (llama3.2:1b) on 73-prompt realistic traffic:

- **Cache hit latency**: 1.4ms median
- **Cache miss latency**: 790.5ms median (backend inference time)
- **Cache hit rate**: 10-11% baseline with Jaccard threshold 0.85

Cache hits are served from memory without backend requests. Hit rate depends on
traffic patterns and similarity threshold tuning.

## Use Cases

1. **Development/Testing**: Avoid repeatedly calling expensive inference APIs
2. **Chatbot FAQ**: Cache common questions with semantic matching
3. **API Cost Reduction**: Reduce backend load for similar queries
4. **Latency Optimization**: Sub-10ms responses for cached prompts

## Limitations

- **In-memory only**: Cache is lost when proxy restarts
- **Single backend**: No multi-backend routing or load balancing
- **No persistence**: No disk-based cache or distributed caching
- **Similarity matching**: Jaccard threshold needs tuning for your use case

## Difference from Benchmark Cache

The proxy mode is **fundamentally different** from the benchmark cache:

| Feature | Proxy Cache | Benchmark Cache |
|---------|-------------|-----------------|
| Clients | **External HTTP clients** (curl, OpenAI SDK, etc.) | Throttle's internal load generator |
| Use case | Production traffic caching | Benchmark acceleration |
| Visibility | All clients benefit from cache | Only Throttle benchmark requests |
| Control | Separate server process | Integrated into benchmark runs |

## Example: Validating Cache Effectiveness

```python
import time
import httpx

proxy_url = "http://localhost:8080/v1/chat/completions"
prompt = "Explain quantum computing in simple terms"

# Measure cache miss
start = time.time()
response1 = httpx.post(proxy_url, json={
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 100
})
miss_time = time.time() - start
print(f"Cache miss: {miss_time*1000:.1f}ms")

# Measure cache hit
start = time.time()
response2 = httpx.post(proxy_url, json={
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 100
})
hit_time = time.time() - start
print(f"Cache hit: {hit_time*1000:.1f}ms")
print(f"Speedup: {miss_time/hit_time:.1f}x")
```

## Next Steps

- Try different similarity thresholds to tune cache effectiveness
- Monitor cache stats to track hit rates
- Test with your production traffic patterns
- Compare cache hit latency vs backend latency

For questions or issues, see the main Throttle documentation.
