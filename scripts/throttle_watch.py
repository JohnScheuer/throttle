#!/usr/bin/env python3
"""
throttle watch — one command, see your vLLM spend in 30 seconds.

Usage:
    python scripts/throttle_watch.py \
        --metrics-url http://localhost:8000/metrics \
        --gpu-rate-per-hour 2.50

    # Measure throughput at an alternative config for 10 minutes:
    python scripts/throttle_watch.py \
        --metrics-url http://localhost:8000/metrics \
        --gpu-rate-per-hour 2.50 \
        --try max_num_seqs=32 \
        --try-duration-minutes 10 \
        --emit-calibration   # opt-in: anonymous throughput point contributed

Standalone — can be run without installing throttle.
Add src/ to PYTHONPATH or install the package first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from throttle.advisor import stream_metrics, snapshot_once, CostSnapshot
except ImportError:
    print(
        "Cannot import throttle.advisor. Run with:\n"
        "  PYTHONPATH=src python scripts/throttle_watch.py ...",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Rendering (thin layer — the module returns JSON, this formats it)
# ---------------------------------------------------------------------------

def _render(snap: CostSnapshot, compact: bool = False) -> None:
    if compact:
        # Single-line for --try summary
        tput = (f"{snap.generation_throughput_toks_per_sec:.0f} tok/s"
                if snap.generation_throughput_toks_per_sec else "—")
        cost = (f"${snap.cost_per_million_tokens:.2f}/Mtok"
                if snap.cost_per_million_tokens else "unavailable")
        fill = (f"{snap.batch_fill:.0%} fill"
                if snap.batch_fill is not None else "fill unknown")
        print(f"  {tput:>14}  {cost:>16}  {fill}")
        return

    print()
    print("=" * 60)
    print(f"  vLLM Cost Snapshot  —  {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    if snap.generation_throughput_toks_per_sec is not None:
        print(f"  Generation throughput : "
              f"{snap.generation_throughput_toks_per_sec:.1f} tok/s")
    else:
        print(f"  Generation throughput : UNAVAILABLE")

    if snap.prompt_throughput_toks_per_sec is not None:
        print(f"  Prompt throughput     : "
              f"{snap.prompt_throughput_toks_per_sec:.1f} tok/s")

    if snap.num_requests_running is not None:
        fill_str = (f"  ({snap.batch_fill:.0%} fill)"
                    if snap.batch_fill is not None else "")
        print(f"  Requests running      : "
              f"{snap.num_requests_running:.0f}{fill_str}")

    if snap.num_requests_waiting is not None:
        print(f"  Requests waiting      : "
              f"{snap.num_requests_waiting:.0f}")

    if snap.gpu_cache_usage_perc is not None:
        print(f"  KV cache usage        : "
              f"{snap.gpu_cache_usage_perc:.0%}")

    print()

    if snap.cost_per_hour is not None:
        print(f"  Cost (GPU)            : ${snap.cost_per_hour:.2f}/hr")
    if snap.cost_per_million_tokens is not None:
        print(f"  Cost per million tok  : ${snap.cost_per_million_tokens:.4f}")

    if snap.refusals:
        print()
        print("  REFUSED:")
        for r in snap.refusals:
            print(f"    [{r['figure']}] {r['reason']}")

    if snap.metrics_unavailable:
        print()
        print(f"  Metrics not exposed   : {', '.join(snap.metrics_unavailable)}")

    print(f"  Observation           : {snap.observation_seconds:.0f}s")
    print()


# ---------------------------------------------------------------------------
# --try: measure throughput at alternative config
# ---------------------------------------------------------------------------

def run_try(
    metrics_url: str,
    gpu_rate: float,
    config_str: str,
    duration_minutes: float,
    max_num_seqs: int | None,
    emit_calibration: bool,
) -> None:
    """
    Scrape metrics for duration_minutes, print throughput summary.
    Does not change any config — user applies the change manually first.
    """
    print(f"\n--try: measuring throughput for {duration_minutes}m")
    print(f"  Config label  : {config_str}")
    print(f"  NOTE: Apply '{config_str}' to your vLLM instance before")
    print(f"        starting this measurement.")
    print()
    print(f"  {'Elapsed':>8}  {'Throughput':>14}  {'Cost':>16}  {'Batch fill'}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*16}  {'-'*12}")

    end_time = time.time() + duration_minutes * 60
    throughputs = []
    interval = 15.0

    for snap in stream_metrics(metrics_url, gpu_rate,
                               interval_seconds=interval,
                               max_num_seqs=max_num_seqs):
        elapsed = duration_minutes * 60 - (end_time - time.time())
        print(f"  {elapsed:>7.0f}s", end="")
        _render(snap, compact=True)

        if snap.generation_throughput_toks_per_sec:
            throughputs.append(snap.generation_throughput_toks_per_sec)

        if time.time() >= end_time:
            break

    if not throughputs:
        print("\n  No throughput data collected. Was the backend serving requests?")
        return

    import statistics
    mean_tput = statistics.mean(throughputs)
    p50_tput = statistics.median(throughputs)

    print()
    print(f"  RESULT ({len(throughputs)} samples over {duration_minutes}m):")
    print(f"    Config        : {config_str}")
    print(f"    Mean throughput : {mean_tput:.1f} tok/s")
    print(f"    P50 throughput  : {p50_tput:.1f} tok/s")
    if gpu_rate and mean_tput > 0:
        cost_mtok = (gpu_rate / 3600 / mean_tput) * 1_000_000
        print(f"    Cost/Mtok       : ${cost_mtok:.4f}")
    print(f"    Basis           : measured ({duration_minutes}m observation)")

    if emit_calibration:
        _emit_calibration_point(config_str, mean_tput, p50_tput,
                                duration_minutes, len(throughputs))


def _emit_calibration_point(
    config_str: str,
    mean_tput: float,
    p50_tput: float,
    duration_minutes: float,
    n_samples: int,
) -> None:
    """
    Emit an anonymous calibration point.
    DISCLOSED: printed to stdout so user sees exactly what would be sent.
    OFF by default — only runs when --emit-calibration is passed.
    Contains: config, measured throughput, sample count.
    Does NOT contain: traffic content, prompts, responses, IP address.
    """
    import platform

    point = {
        "schema": "throttle_calibration_v1",
        "config": config_str,
        "mean_throughput_toks_per_sec": round(mean_tput, 2),
        "p50_throughput_toks_per_sec": round(p50_tput, 2),
        "observation_minutes": duration_minutes,
        "n_samples": n_samples,
        "python_version": platform.python_version(),
        # GPU info intentionally omitted until user opts in with --include-gpu-info
    }

    print()
    print("  CALIBRATION POINT (--emit-calibration is ON)")
    print("  This would be contributed anonymously:")
    print(json.dumps(point, indent=4))
    print()
    print("  NOTE: Actual submission not yet implemented.")
    print("  This output shows exactly what would be sent.")
    print("  No data has left your machine.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metrics-url",
        default="http://localhost:8000/metrics",
        help="vLLM /metrics endpoint (default: http://localhost:8000/metrics)",
    )
    p.add_argument(
        "--gpu-rate-per-hour",
        type=float,
        required=True,
        metavar="DOLLARS",
        help="GPU cost in $/hr. Required — no default is honest.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="Scrape interval in seconds (default: 15)",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        metavar="N",
        help="vLLM max_num_seqs for batch fill computation. "
             "Read from vLLM config if exposed; otherwise supply here.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON snapshots instead of formatted text",
    )
    p.add_argument(
        "--try",
        dest="try_config",
        default=None,
        metavar="KEY=VALUE",
        help="Measure throughput at an alternative config for --try-duration minutes. "
             "Apply the config to vLLM manually first, then run this.",
    )
    p.add_argument(
        "--try-duration-minutes",
        type=float,
        default=10.0,
        metavar="MINUTES",
        help="Duration for --try measurement (default: 10 minutes)",
    )
    p.add_argument(
        "--emit-calibration",
        action="store_true",
        help=(
            "OPT-IN: Print the anonymous calibration point that --try would "
            "contribute (model, config, measured throughput). "
            "No data is sent — output shows exactly what would be submitted. "
            "Submission not yet implemented."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.try_config:
        run_try(
            metrics_url=args.metrics_url,
            gpu_rate=args.gpu_rate_per_hour,
            config_str=args.try_config,
            duration_minutes=args.try_duration_minutes,
            max_num_seqs=args.max_num_seqs,
            emit_calibration=args.emit_calibration,
        )
        return

    print(f"Watching {args.metrics_url} every {args.interval:.0f}s")
    print(f"GPU rate: ${args.gpu_rate_per_hour:.2f}/hr")
    print("Press Ctrl+C to stop.")

    try:
        for snap in stream_metrics(
            args.metrics_url,
            args.gpu_rate_per_hour,
            interval_seconds=args.interval,
            max_num_seqs=args.max_num_seqs,
        ):
            if args.json:
                print(snap.to_json())
                sys.stdout.flush()
            else:
                _render(snap)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
