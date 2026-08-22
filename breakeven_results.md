## Break-Even Hit Rate: Throttle (in-process) vs. GPTCache (remote vector DB)

- **ONNX Embedding Latency:** 4.66ms (measured locally)
- **Remote Vector-DB Lookup:** 30.00ms (GPTCache literature baseline)

### 🎯 Direct Comparison (Headline):
> **At 500 entries (300ms backend):**
> - Throttle Break-Even: **1.61%** (Honest: Jaccard scan + ONNX embedding on miss)
> - GPTCache Break-Even: **10.00%** (Remote Vector-DB lookup)
> - **Result:** Throttle is **6.2× more efficient** — saves wall-clock time starting at ~1.6% hit rate.
>
> *(At 500ms backend: Throttle **0.97%** vs. GPTCache **6.00%** → **6.2× more efficient**)*

| Cache Size | Jaccard Scan | + ONNX (miss) | h_be @300ms backend | h_be @500ms backend | Jaccard-only h_be @300 | GPTCache h_be @300 | GPTCache h_be @500 | Viable? |
|-----------|-------------|---------------|--------------------|--------------------|----------------------|-------------------|-------------------|---------|
|       500 |      0.25ms |        4.90ms |             1.61% |             0.97% |               0.08% |           10.00% |            6.00% | ✅ |
|     1,000 |      0.52ms |        5.17ms |             1.70% |             1.02% |               0.17% |           10.00% |            6.00% | ✅ |
|     5,000 |      2.93ms |        7.59ms |             2.49% |             1.50% |               0.98% |           10.00% |            6.00% | ✅ |
|    10,000 |      5.94ms |       10.60ms |             3.48% |             2.10% |               1.98% |           10.00% |            6.00% | ✅ |
|    20,000 |     11.71ms |       16.36ms |             5.37% |             3.24% |               3.90% |           10.00% |            6.00% | ⚠️ >10ms |
|    50,000 |     27.28ms |       31.93ms |            10.48% |             6.33% |               9.09% |           10.00% |            6.00% | ⚠️ >10ms |

**Key Takeaways:**
- **h_be (Break-Even):** Minimum hit rate required for caching to save wall-clock time.
- **Honest Cost:** Includes the mandatory ONNX embedding pass on cache misses.
- **Crossover Frontier:** Throttle remains strictly superior to remote vector DBs up to **~20,000 entries**.
- Beyond 20,000 entries, the linear O(N) Jaccard scan exceeds 10ms, marking the boundary where index structures (HNSW/Vector DB) become justified.
