"""Tests for throttle diagnose pre-flight regime classification."""

import unittest
from pathlib import Path

from throttle.cli import _build_config, build_parser
from throttle.diagnose import (
    DIAGNOSE_ARTIFACT_TYPE,
    ProbeMetrics,
    _build_diagnose_report,
    _classify_regime,
    _extract_probe_metrics,
)
from throttle.models import EndpointConfig, RunConfig, SafetyLimits


class TestDiagnose(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_config = RunConfig(
            mode="smoke",
            backend="native",
            model="test-model",
            endpoint=EndpointConfig(url="http://127.0.0.1:8000/v1", api_key="dummy"),
            limits=SafetyLimits(max_requests=100),
        )

    def test_cli_parser_diagnose_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "diagnose",
                "--model",
                "test-model",
                "--url",
                "http://127.0.0.1:8000/v1",
            ]
        )
        self.assertEqual(args.command, "diagnose")
        self.assertEqual(args.model, "test-model")
        self.assertIsNone(args.concurrency)
        self.assertEqual(args.backend, "native")
        self.assertEqual(args.output, Path("throttle-diagnose.json"))

        # Check that _build_config resolves default concurrency [1, 4, 8] and safety limits
        config, prompts, warmups = _build_config(parser, args, resolve_key=False)
        concurrencies = [int(c.value) for c in config.conditions]
        self.assertEqual(concurrencies, [1, 4, 8])
        self.assertEqual(config.limits.max_elapsed_seconds, 60.0)
        self.assertEqual(config.requests_per_block, 20)
        self.assertEqual(config.warmup_requests_per_condition, 3)

    def test_classify_dispatch_bound(self) -> None:
        # TTFT dominance in range (0.6, 0.70], poor scaling efficiency
        # avg_ttft=65, avg_tpot=35 -> ttft_dominance = 0.65
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=60.0, tpot_mean_ms=30.0, throughput_tokens_per_sec=20.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=65.0, tpot_mean_ms=35.0, throughput_tokens_per_sec=30.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=70.0, tpot_mean_ms=40.0, throughput_tokens_per_sec=35.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "dispatch-bound")
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_classify_compute_bound(self) -> None:
        # Linear scaling, stable low TPOT CV
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=20.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=50.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=22.0, tpot_mean_ms=15.2, throughput_tokens_per_sec=190.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=25.0, tpot_mean_ms=15.5, throughput_tokens_per_sec=360.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "compute-bound")
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_classify_orchestration_bound(self) -> None:
        # Sublinear scaling (< 0.6), low TTFT dominance (< 0.4)
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=15.0, tpot_mean_ms=35.0, throughput_tokens_per_sec=50.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=16.0, tpot_mean_ms=36.0, throughput_tokens_per_sec=90.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=18.0, tpot_mean_ms=38.0, throughput_tokens_per_sec=110.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "orchestration-bound")
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_classify_memory_bound(self) -> None:
        # High TTFT dominance (> 0.70), poor scaling (< 0.45)
        # avg_ttft=176.67, avg_tpot=10 -> ttft_dominance = 0.946 > 0.70
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=150.0, tpot_mean_ms=10.0, throughput_tokens_per_sec=50.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=180.0, tpot_mean_ms=10.0, throughput_tokens_per_sec=70.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=200.0, tpot_mean_ms=10.0, throughput_tokens_per_sec=80.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "memory-bound")
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_classify_dispatch_bound_at_boundary(self) -> None:
        # Boundary test: ttft_dominance = exactly 0.70 should be dispatch-bound (inclusive upper bound)
        # avg_ttft=70, avg_tpot=30 -> ttft_dominance = 0.70
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=68.0, tpot_mean_ms=29.0, throughput_tokens_per_sec=20.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=70.0, tpot_mean_ms=30.0, throughput_tokens_per_sec=30.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=72.0, tpot_mean_ms=31.0, throughput_tokens_per_sec=35.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "dispatch-bound")
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_classify_mixed(self) -> None:
        # Doesn't match any specific regime criteria - moderate values across the board
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=50.0, tpot_mean_ms=50.0, throughput_tokens_per_sec=50.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=52.0, tpot_mean_ms=52.0, throughput_tokens_per_sec=110.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=55.0, tpot_mean_ms=55.0, throughput_tokens_per_sec=150.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "mixed")
        self.assertEqual(result.confidence, 0.50)

    def test_classify_inconclusive_high_errors(self) -> None:
        # More than 50% error rate
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=20.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=50.0, valid_count=2, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=22.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=20.0, valid_count=1, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "inconclusive")
        self.assertIn("high_error_rate", result.quality_issues)

    def test_classify_inconclusive_insufficient_samples(self) -> None:
        # Less than 10 total valid samples
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=20.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=50.0, valid_count=3, attempted_count=3),
            ProbeMetrics(concurrency=4, ttft_mean_ms=22.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=190.0, valid_count=5, attempted_count=5),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "inconclusive")
        self.assertIn("insufficient_valid_samples", result.quality_issues)

    def test_classify_inconclusive_high_variance(self) -> None:
        # High TPOT variance (CV > 1.0) indicating bimodal/unstable behavior
        # Using values: 5, 10, 100 ms -> mean=38.33, stdev=53.47, CV=1.395 > 1.0
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=20.0, tpot_mean_ms=5.0, throughput_tokens_per_sec=50.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=22.0, tpot_mean_ms=10.0, throughput_tokens_per_sec=190.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=8, ttft_mean_ms=25.0, tpot_mean_ms=100.0, throughput_tokens_per_sec=360.0, valid_count=20, attempted_count=20),
        ]
        result = _classify_regime(probes)
        self.assertEqual(result.classification, "inconclusive")
        self.assertIn("high_tpot_variance", result.quality_issues)

    def test_report_schema_compliance(self) -> None:
        probes = [
            ProbeMetrics(concurrency=1, ttft_mean_ms=20.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=50.0, valid_count=20, attempted_count=20),
            ProbeMetrics(concurrency=4, ttft_mean_ms=22.0, tpot_mean_ms=15.0, throughput_tokens_per_sec=180.0, valid_count=20, attempted_count=20),
        ]
        regime = _classify_regime(probes)
        dummy_run_report = {
            "status": "complete",
            "conditions": [
                {"condition": {"value": 1}, "request_counts": {"valid": 20, "attempted": 20}},
                {"condition": {"value": 4}, "request_counts": {"valid": 20, "attempted": 20}},
            ],
        }
        report = _build_diagnose_report(dummy_run_report, regime, self.dummy_config)
        self.assertEqual(report["schema_version"], "2.0")
        self.assertEqual(report["artifact_type"], DIAGNOSE_ARTIFACT_TYPE)
        self.assertEqual(report["mode"], "smoke")
        self.assertFalse(report["decision_eligible"])
        self.assertIn("regime", report)
        self.assertIn("recommended_config_dimensions", report)


if __name__ == "__main__":
    unittest.main()
