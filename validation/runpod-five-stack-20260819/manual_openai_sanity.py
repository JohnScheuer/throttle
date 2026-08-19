#!/usr/bin/env python3
"""Small standalone OpenAI-compatible streaming load check (not Throttle)."""

import argparse
import asyncio
import json
import os
import statistics
import time

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompts", required=True)
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    completion_tokens = 0
    errors: list[str] = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(
        timeout=120,
        follow_redirects=False,
        trust_env=False,
        headers={"Authorization": f"Bearer {os.environ[args.api_key_env]}"},
    ) as client:
        async def one(index: int) -> None:
            nonlocal completion_tokens
            prompt = rows[index % len(rows)]
            messages = prompt.get("messages")
            if messages is None:
                messages = [{"role": "user", "content": prompt["prompt"]}]
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            async with semaphore:
                started = time.perf_counter()
                tokens = None
                try:
                    async with client.stream("POST", args.url, json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            event = json.loads(line[6:])
                            usage = event.get("usage")
                            if usage:
                                tokens = usage.get("completion_tokens")
                    elapsed = time.perf_counter() - started
                    if not isinstance(tokens, int) or tokens <= 0:
                        raise ValueError(f"missing positive completion usage: {tokens!r}")
                    async with lock:
                        latencies.append(elapsed)
                        completion_tokens += tokens
                except Exception as exc:
                    async with lock:
                        errors.append(f"request {index}: {type(exc).__name__}: {exc}")

        wall_started = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall = time.perf_counter() - wall_started

    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1)) if ordered else 0
    result = {
        "client": "standalone httpx streaming script (not Throttle)",
        "concurrency": args.concurrency,
        "requested": args.requests,
        "valid": len(latencies),
        "errors": errors,
        "wall_seconds": wall,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / wall if wall else None,
        "latency_ms_mean": statistics.mean(latencies) * 1000 if latencies else None,
        "latency_ms_p95_nearest_rank": ordered[p95_index] * 1000 if ordered else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
