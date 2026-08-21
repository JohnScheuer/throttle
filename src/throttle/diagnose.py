"""Pre-flight bottleneck regime classification for Throttle v2.

Runs lightweight client-side probes to classify whether the dominant serving
bottleneck is dispatch-bound, orchestration-bound, compute-bound, or memory-bound
before running full benchmark or golden protocols.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .benchmark import SCHEMA_VERSION, DIAGNOSE_ARTIFACT_TYPE, RunProgress, run_native
from .models import RunConfig

REGIME_RECOMMENDATIONS: dict[str, dict[str, Any]] = {
    "dispatch-bound": {
        "recommended": [
            "cuda_graph_capture",
            "batch_size",
            "kernel_fusion_flags",
        ],
        "not_recommended": [
            "max_num_seqs (dispatch is the bottleneck, not scheduling)",
            "quantization (compute is not the limiter)",
        ],
        "next_step": "Enable CUDA Graphs or adjust launch batching, then re-run diagnose.",
    },
    "orchestration-bound": {
        "recommended": [
            "scheduler_config",
            "request_batching_strategy",
            "python_vs_cpp_runtime",
        ],
        "not_recommended": [
            "kernel_optimization (GPU is not the bottleneck)",
        ],
        "next_step": "Profile host-side orchestration overhead or C++ serving core.",
    },
    "compute-bound": {
        "recommended": [
            "max_num_seqs",
            "max_num_batched_tokens",
            "quantization (INT8/INT4)",
            "kernel_fusion",
        ],
        "not_recommended": [
            "cuda_graph_capture (dispatch is not the bottleneck)",
        ],
        "next_step": "Run throttle golden on max_num_seqs sweep.",
    },
    "memory-bound": {
        "recommended": [
            "kv_cache_block_size",
            "swap_policy",
            "prefix_caching",
            "max_model_len",
        ],
        "not_recommended": [
            "max_num_seqs (memory capacity/bandwidth is the constraint)",
        ],
        "next_step": "Reduce max_model_len or enable prefix caching.",
    },
    "mixed": {
        "recommended": [
            "max_num_seqs",
            "max_num_batched_tokens",
        ],
        "not_recommended": [],
        "next_step": "Run an exploratory throttle benchmark sweep to isolate the dominant layer.",
    },
    "inconclusive": {
        "recommended": [],
        "not_recommended": [],
        "next_step": "Resolve network/load instability before running diagnose again.",
    },
}


@dataclass
class ProbeMetrics:
    """Aggregated timing metrics extracted from one probed concurrency level."""

    concurrency: int
    ttft_mean_ms: float | None = None
    ttft_p95_ms: float | None = None
    tpot_mean_ms: float | None = None
    tpot_p95_ms: float | None = None
    throughput_tokens_per_sec: float | None = None
    requests_per_sec: float | None = None
    valid_count: int = 0
    attempted_count: int = 0


@dataclass
class RegimeResult:
    """Classification output with supporting evidence."""

    classification: str
    confidence: float
    evidence: dict[str, float]
    quality_issues: list[str] = field(default_factory=list)


def _extract_probe_metrics(report: Mapping[str, Any]) -> list[ProbeMetrics]:
    """Extract per-condition metrics from a native run report."""
    results: list[ProbeMetrics] = []
    for item in report.get("conditions", []):
        condition = item.get("condition", {})
        concurrency = int(condition.get("value", 1))

        # Use diagnostic_metrics as fallback if metrics is None (e.g. for short runs)
        metrics = item.get("metrics") or item.get("diagnostic_metrics") or {}
        counts = item.get("request_counts", {})

        ttft = metrics.get("ttft_ms") or {}
        tpot = metrics.get("tpot_ms") or {}

        throughput = metrics.get("output_tokens_per_second")
        if throughput is None and metrics.get("block_mean_output_tokens_per_second") is not None:
            throughput = metrics.get("block_mean_output_tokens_per_second")

        probe = ProbeMetrics(
            concurrency=concurrency,
            ttft_mean_ms=float(ttft["mean"]) if ttft.get("mean") is not None else None,
            ttft_p95_ms=float(ttft["p95"]) if ttft.get("p95") is not None else None,
            tpot_mean_ms=float(tpot["mean"]) if tpot.get("mean") is not None else None,
            tpot_p95_ms=float(tpot["p95"]) if tpot.get("p95") is not None else None,
            throughput_tokens_per_sec=float(throughput) if throughput is not None else None,
            requests_per_sec=float(metrics["requests_per_second"])
            if metrics.get("requests_per_second") is not None
            else None,
            valid_count=int(counts.get("valid", 0)),
            attempted_count=int(counts.get("attempted", 0)),
        )
        results.append(probe)
    return results


def _classify_regime(probes: list[ProbeMetrics]) -> RegimeResult:
    """Classify the dominant bottleneck regime using client-side timing heuristics."""
    quality_issues: list[str] = []

    total_valid = sum(p.valid_count for p in probes)
    total_attempted = sum(p.attempted_count for p in probes)

    if total_valid < 10:
        quality_issues.append("insufficient_valid_samples")

    if total_attempted > 0 and (total_attempted - total_valid) / total_attempted > 0.5:
        quality_issues.append("high_error_rate")

    valid_probes = [p for p in probes if p.valid_count > 0 and p.throughput_tokens_per_sec is not None]
    # Sort by concurrency to ensure baseline is lowest and peak is highest
    valid_probes.sort(key=lambda p: p.concurrency)

    if len(valid_probes) < 2:
        if "insufficient_valid_samples" not in quality_issues:
            quality_issues.append("insufficient_valid_concurrency_levels")
        return RegimeResult(
            classification="inconclusive",
            confidence=0.0,
            evidence={},
            quality_issues=quality_issues,
        )

    # 1. Throughput Scaling Efficiency (between lowest and highest concurrency)
    baseline_p = valid_probes[0]
    peak_p = valid_probes[-1]
    baseline_tp = baseline_p.throughput_tokens_per_sec or 0.001
    peak_tp = peak_p.throughput_tokens_per_sec or 0.001
    concurrency_ratio = peak_p.concurrency / max(baseline_p.concurrency, 1)

    scaling_efficiency = (peak_tp / max(baseline_tp, 0.001)) / max(concurrency_ratio, 1.0)

    # 2. TTFT vs TPOT Latency Breakdown
    all_ttft = [p.ttft_mean_ms for p in valid_probes if p.ttft_mean_ms is not None]
    all_tpot = [p.tpot_mean_ms for p in valid_probes if p.tpot_mean_ms is not None]

    avg_ttft = statistics.mean(all_ttft) if all_ttft else 0.0
    avg_tpot = statistics.mean(all_tpot) if all_tpot else 0.0
    total_latency = avg_ttft + avg_tpot
    ttft_dominance = avg_ttft / total_latency if total_latency > 0 else 0.0

    # 3. TPOT Stability (Coefficient of Variation across concurrency levels)
    if len(all_tpot) >= 2 and statistics.mean(all_tpot) > 0:
        tpot_cv = statistics.stdev(all_tpot) / statistics.mean(all_tpot)
    else:
        tpot_cv = 0.0

    # 4. Check for high variance / bimodal jitter
    if tpot_cv > 1.0:
        quality_issues.append("high_tpot_variance")

    if len(all_ttft) >= 2 and statistics.mean(all_ttft) > 0:
        ttft_cv = statistics.stdev(all_ttft) / statistics.mean(all_ttft)
        if ttft_cv > 1.0:
            quality_issues.append("high_ttft_variance")

    evidence = {
        "throughput_scaling_efficiency": round(scaling_efficiency, 3),
        "ttft_dominance_ratio": round(ttft_dominance, 3),
        "tpot_stability_cv": round(tpot_cv, 3),
    }

    if quality_issues:
        return RegimeResult(
            classification="inconclusive",
            confidence=0.0,
            evidence=evidence,
            quality_issues=quality_issues,
        )

    # Regime Classification Tree
    if scaling_efficiency < 0.5 and ttft_dominance > 0.6:
        return RegimeResult("dispatch-bound", 0.80, evidence)
    if scaling_efficiency < 0.6 and ttft_dominance < 0.4:
        return RegimeResult("orchestration-bound", 0.70, evidence)
    if tpot_cv < 0.25 and scaling_efficiency > 0.70:
        return RegimeResult("compute-bound", 0.85, evidence)
    if ttft_dominance > 0.70 and scaling_efficiency < 0.45:
        return RegimeResult("memory-bound", 0.75, evidence)

    return RegimeResult("mixed", 0.50, evidence)


def _build_diagnose_report(
    report: Mapping[str, Any],
    regime: RegimeResult,
    config: RunConfig,
) -> dict[str, Any]:
    """Build the final schema-2.0 diagnose report."""
    recs = REGIME_RECOMMENDATIONS.get(
        regime.classification, REGIME_RECOMMENDATIONS["mixed"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": DIAGNOSE_ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke",
        "status": report.get("status", "complete"),
        "decision_eligible": False,
        "regime": {
            "classification": regime.classification,
            "confidence": regime.confidence,
            "evidence": regime.evidence,
            "quality_issues": regime.quality_issues,
        },
        "recommended_config_dimensions": recs["recommended"],
        "not_recommended": recs["not_recommended"],
        "next_step": recs["next_step"],
        "probe_summary": {
            "concurrency_levels": [
                c.get("condition", {}).get("value") for c in report.get("conditions", [])
            ],
            "total_valid_requests": sum(
                c.get("request_counts", {}).get("valid", 0)
                for c in report.get("conditions", [])
            ),
            "total_attempted_requests": sum(
                c.get("request_counts", {}).get("attempted", 0)
                for c in report.get("conditions", [])
            ),
        },
        "disclaimer": (
            "This diagnose report is a pre-flight heuristic classification, "
            "not a measured performance comparison or production claim. Use "
            "`throttle golden` for decision-grade evidence."
        ),
    }


def print_diagnose(report: Mapping[str, Any], output: Path) -> None:
    """Render the diagnose results clearly to the terminal."""
    regime = report.get("regime", {})
    classification = regime.get("classification", "unknown")
    confidence = regime.get("confidence", 0.0)

    print()
    print("Throttle DIAGNOSE — PRE-FLIGHT BOTTLENECK CLASSIFICATION")
    print(f"Classification: {classification.upper()} (confidence: {confidence:.0%})")

    evidence = regime.get("evidence", {})
    if evidence:
        print("Evidence heuristics:")
        for k, v in evidence.items():
            print(f"  • {k}: {v}")

    quality = regime.get("quality_issues", [])
    if quality:
        print(f"Quality flags: {', '.join(quality)}")

    recs = report.get("recommended_config_dimensions", [])
    if recs:
        print("Recommended configuration dimensions to tune/test:")
        for r in recs:
            print(f"  [+] {r}")

    not_recs = report.get("not_recommended", [])
    if not_recs:
        print("Not recommended for this regime:")
        for r in not_recs:
            print(f"  [-] {r}")

    print(f"Next step: {report.get('next_step', 'N/A')}")
    print(f"Sanitized JSON artifact: {output}")
    print()


def handle_diagnose(
    parser: Any,
    args: Any,
    build_config_fn: Callable,
    write_fn: Callable,
) -> int:
    """Main CLI execution handler for `throttle diagnose`."""
    # Enforce safe pre-flight defaults if not explicitly passed
    if args.max_elapsed_seconds is None:
        args.max_elapsed_seconds = 60.0
    if args.max_requests == 10_000:
        args.max_requests = 200
    if args.max_errors == 1:
        args.max_errors = 3
    if args.max_estimated_spend == 3.0:
        args.max_estimated_spend = 0.50

    if args.blocks is None:
        args.blocks = 1
    if args.requests_per_block is None and args.block_seconds is None:
        args.requests_per_block = 20
    if args.warmup_requests is None:
        args.warmup_requests = 3

    config, prompts, warmup_prompts = build_config_fn(parser, args, resolve_key=True)
    progress = RunProgress()

    try:
        report = asyncio.run(
            run_native(config, prompts, warmup_prompts, progress=progress)
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Diagnose cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Diagnose probe execution failed: {exc}", file=sys.stderr)
        return 1

    probes = _extract_probe_metrics(report)
    regime = _classify_regime(probes)
    diagnose_report = _build_diagnose_report(report, regime, config)

    try:
        write_fn(diagnose_report, args.output)
    except (OSError, TypeError, ValueError, OverflowError):
        print("Diagnose artifact could not be written.", file=sys.stderr)
        return 1

    print_diagnose(diagnose_report, args.output)

    if regime.classification in {"inconclusive", "mixed"}:
        return 3
    return 0