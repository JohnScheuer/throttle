"""Benchmark cache hit rate and speedup vs no-cache baseline.

Usage:
    python tests/benchmark_cache.py
    python tests/benchmark_cache.py --repeat 5 --embeddings
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Add src and tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from throttle.cache import SimilarityCache
from mock_backend import mock_chat_completion


# Workload with exact duplicates, near-duplicates, paraphrases, and unique queries
DEFAULT_WORKLOAD: list[str] = [
    "What is the capital of France?",
    "What is the capital of france?",  # Exact after normalization
    "what's the capital of France",  # Contraction + near-duplicate
    "Tell me the capital of France",  # Paraphrase
    "What is the capital of Japan?",
    "capital of Japan?",  # Near-duplicate
    "How do I reset my password?",
    "password reset steps",  # Paraphrase, lower lexical overlap
    "I forgot my password, how do I reset it?",  # Paraphrase
    "What is the largest planet in the solar system?",
    "largest planet in solar system",  # Near-duplicate
    "How can I cancel my subscription?",
    "cancel subscription steps",  # Paraphrase
    "What's the speed of light?",
    "speed of light value",  # Near-duplicate
    "What is the refund policy?",
    "how do refunds work",  # Paraphrase
    "Explain the difference between a Python list and a tuple",
    "python list vs tuple differences",  # Near-duplicate
    "What is machine learning?",
    "define machine learning",  # Paraphrase
    "What is a neural network?",
    "explain neural networks",  # Paraphrase
    "How does TCP work?",
    "explain how TCP works",  # Near-duplicate
    "What is the weather like on Mars?",  # Unique, should always miss
    "Recommend a good sci-fi book",  # Unique, should always miss
    "What is the capital of India?",
    "capital of india",  # Exact after normalization
]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_with_cache(
    queries: list[str], cache: SimilarityCache
) -> tuple[float, list[float], list[float]]:
    """Run queries through cache. Returns (total_ms, lookup_latencies, e2e_latencies)."""
    lookup_latencies = []
    e2e_latencies = []

    start = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        result = cache.get(q)
        lookup_time = (time.perf_counter() - t0) * 1000

        if result is None:
            # Cache miss: forward to backend
            response = mock_chat_completion(q, simulate_latency=True)
            cache.put(q, response)
            e2e_time = (time.perf_counter() - t0) * 1000
        else:
            # Cache hit: no backend call
            e2e_time = lookup_time

        lookup_latencies.append(lookup_time)
        e2e_latencies.append(e2e_time)

    total_ms = (time.perf_counter() - start) * 1000
    return total_ms, lookup_latencies, e2e_latencies


def run_without_cache(queries: list[str]) -> tuple[float, list[float]]:
    """Run queries directly against backend. Returns (total_ms, latencies)."""
    latencies = []
    start = time.perf_counter()

    for q in queries:
        t0 = time.perf_counter()
        mock_chat_completion(q, simulate_latency=True)
        latencies.append((time.perf_counter() - t0) * 1000)

    total_ms = (time.perf_counter() - start) * 1000
    return total_ms, latencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark semantic cache effectiveness.")
    parser.add_argument(
        "--repeat", type=int, default=3, help="Repeat workload N times to simulate sustained traffic"
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42, help="Seed for shuffling; use -1 to disable"
    )
    parser.add_argument(
        "--jaccard-threshold", type=float, default=0.85, help="Jaccard similarity threshold"
    )
    parser.add_argument(
        "--embeddings", action="store_true", help="Enable ONNX embeddings tier (requires deps)"
    )
    parser.add_argument(
        "--embedding-threshold", type=float, default=0.95, help="Embedding cosine threshold"
    )
    parser.add_argument("--out", type=str, default="cache_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    # Build workload
    workload = DEFAULT_WORKLOAD * args.repeat
    if args.shuffle_seed != -1:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(workload)

    print(f"Workload size: {len(workload)} queries")
    print(f"Unique prompts: {len(set(workload))}")
    print(f"Cache config: Jaccard threshold={args.jaccard_threshold}, Embeddings={args.embeddings}\n")

    # Run with cache
    cache = SimilarityCache(
        similarity_threshold=args.jaccard_threshold,
        enable_embeddings=args.embeddings,
        embedding_threshold=args.embedding_threshold,
        ttl_seconds=7200.0,
        max_size=10000,
    )

    cached_total_ms, lookup_latencies, cached_e2e_latencies = run_with_cache(workload, cache)

    # Run without cache (baseline)
    random.seed(7)  # Keep baseline latency distribution comparable
    baseline_total_ms, baseline_latencies = run_without_cache(workload)

    # Build report
    metrics = cache.metrics
    hit_rate = metrics.hits / (metrics.hits + metrics.misses) if (metrics.hits + metrics.misses) > 0 else 0.0

    report = {
        "workload": {
            "size": len(workload),
            "unique": len(set(workload)),
            "repeat": args.repeat,
        },
        "cache_config": {
            "jaccard_threshold": args.jaccard_threshold,
            "embeddings_enabled": args.embeddings,
            "embedding_threshold": args.embedding_threshold if args.embeddings else None,
        },
        "with_cache": {
            "total_wall_time_ms": round(cached_total_ms, 2),
            "hits": metrics.hits,
            "misses": metrics.misses,
            "hit_rate": round(hit_rate, 4),
            "hit_breakdown": {
                "exact": metrics.exact_hits,
                "jaccard": metrics.lexical_hits,
                "embeddings": metrics.embedding_hits,
            },
            "lookup_latency_ms": {
                "p50": round(percentile(lookup_latencies, 50), 4),
                "p95": round(percentile(lookup_latencies, 95), 4),
                "p99": round(percentile(lookup_latencies, 99), 4),
                "mean": round(sum(lookup_latencies) / len(lookup_latencies), 4),
            },
            "end_to_end_latency_ms": {
                "p50": round(percentile(cached_e2e_latencies, 50), 4),
                "p95": round(percentile(cached_e2e_latencies, 95), 4),
                "p99": round(percentile(cached_e2e_latencies, 99), 4),
                "mean": round(sum(cached_e2e_latencies) / len(cached_e2e_latencies), 4),
            },
        },
        "without_cache_baseline": {
            "total_wall_time_ms": round(baseline_total_ms, 2),
            "latency_ms": {
                "p50": round(percentile(baseline_latencies, 50), 4),
                "p95": round(percentile(baseline_latencies, 95), 4),
                "p99": round(percentile(baseline_latencies, 99), 4),
                "mean": round(sum(baseline_latencies) / len(baseline_latencies), 4),
            },
        },
        "speedup": {
            "wall_time_x": round(baseline_total_ms / cached_total_ms, 2) if cached_total_ms > 0 else None,
        },
    }

    # Print summary
    print(json.dumps(report, indent=2))

    # Write output
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n✓ Full report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
