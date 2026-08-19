from __future__ import annotations

import json
import math
import random
import re
import socket
import subprocess
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

import throttle.bottleneck_analysis as bottleneck_module
from throttle.bottleneck_analysis import (
    ANALYSIS_ARTIFACT_TYPE,
    ANALYSIS_SCHEMA_VERSION,
    HIGH_KV_FRACTION,
    LOW_KV_FRACTION,
    MATERIAL_QUEUE_SHARE,
    MAX_AVERAGE_SAMPLE_SPACING_SECONDS,
    MIN_ELAPSED_SECONDS,
    MIN_FINISHED_REQUESTS,
    MIN_OBSERVATIONS,
    BottleneckAnalysis,
    BottleneckAnalysisError,
    BottleneckContext,
    BottleneckSignals,
    BottleneckSuggestion,
    analyze_bottleneck,
)
from throttle.server_metrics import (
    MAX_OBSERVATIONS,
    MetricsWindow,
    derive_metrics_window,
    parse_vllm_metrics,
)


_PRIVATE_LABEL = "private-model-api-key-do-not-persist"
_SECOND_PRIVATE_LABEL = "different-private-model-and-engine"
_QUALITY_REASONS = frozenset(
    {
        "counter_rate_inconsistent",
        "effective_flags_not_runtime_verified",
        "exporter_request_count_mismatch",
        "finish_reason_evidence_not_clean",
        "gauge_exceeds_offered_concurrency",
        "latency_observation_count_mismatch",
        "latency_metrics_inconsistent",
        "max_num_batched_tokens_below_max_num_seqs",
        "metric_scope_not_single_consistent",
        "missing_required_metric",
        "nonpositive_output_evidence",
        "output_token_count_below_finished_requests",
        "required_metric_series_count_not_one",
        "required_metric_source_missing",
        "running_exceeds_current_max_num_seqs",
        "sample_spacing_too_sparse",
        "throttle_requests_failed",
        "too_few_finished_requests",
        "too_few_observations",
        "traffic_scope_not_exclusive",
        "window_too_short",
    }
)
_NO_SUGGESTION_REASONS = frozenset(
    {
        "configured_limit_not_exercised",
        "kv_headroom_not_clear",
        "legacy_swap_pressure_observed",
        "no_higher_candidate_within_verified_bounds",
        "no_lower_candidate",
        "no_material_queue",
        "pressure_signal_incomplete",
        "queue_and_kv_signals_conflict",
        "queue_share_below_policy_threshold",
    }
)


def _number(value: int | float) -> str:
    if type(value) is int:
        return str(value)
    return format(value, ".17g")


def _line(
    name: str,
    value: int | float,
    *,
    model: str = _PRIVATE_LABEL,
    engine: str | None = None,
    finished_reason: str | None = None,
) -> str:
    labels = [f'model_name="{model}"']
    if engine is not None:
        labels.append(f'engine="{engine}"')
    if finished_reason is not None:
        labels.append(f'finished_reason="{finished_reason}"')
    return f"{name}{{{','.join(labels)}}} {_number(value)}"


def _derived_window(
    *,
    elapsed_seconds: float = 8.0,
    observations: int = 5,
    requests_finished: int = 40,
    output_tokens: int = 400,
    preemptions: int = 0,
    running: int = 8,
    waiting: int = 4,
    swapped: int | None = 0,
    kv_fraction: float = LOW_KV_FRACTION,
    mean_queue_time_ms: float = 200.0,
    mean_ttft_ms: float = 300.0,
    mean_e2e_ms: float = 1_000.0,
    output_series: int = 1,
    output_series_discriminator: str = "engine",
    request_series: int = 1,
    request_finish_reasons: tuple[str | None, ...] | None = None,
    omit: frozenset[str] = frozenset(),
    latency_observations: dict[str, int] | None = None,
    scope_mismatch_family: str | None = None,
) -> MetricsWindow:
    """Build a real collector window from public parser/deriver APIs."""

    if observations < 2:
        raise ValueError("test fixture requires at least two observations")
    intervals = observations - 1
    snapshots = []
    latency_observations = latency_observations or {}

    def scope(logical_name: str) -> dict[str, str]:
        if logical_name == scope_mismatch_family:
            return {
                "model": _SECOND_PRIVATE_LABEL,
                "engine": "private-worker-z",
            }
        return {"model": _PRIVATE_LABEL}

    for index in range(observations):
        request_delta = requests_finished * index // intervals
        output_delta = output_tokens * index // intervals
        preemption_delta = preemptions * index // intervals
        lines: list[str] = []
        if "requests" not in omit:
            reasons = request_finish_reasons
            if reasons is None:
                reasons = (
                    ("stop", "length")
                    if request_series == 2
                    else ("stop",)
                )
            allocated = 0
            for reason_index, reason in enumerate(reasons):
                value = (
                    request_delta - allocated
                    if reason_index == len(reasons) - 1
                    else request_delta // len(reasons)
                )
                allocated += value
                reason_kwargs = (
                    {} if reason is None else {"finished_reason": reason}
                )
                lines.append(
                    _line(
                        "vllm:request_success_total",
                        value,
                        **reason_kwargs,
                        **scope("requests"),
                    )
                )
        if "output" not in omit:
            if output_series == 1:
                lines.append(
                    _line(
                        "vllm:generation_tokens_total",
                        output_delta,
                        **scope("output"),
                    )
                )
            else:
                first_extra = (
                    {"engine": "worker-a"}
                    if output_series_discriminator == "engine"
                    else {"finished_reason": "variant-a"}
                )
                second_extra = (
                    {"engine": "worker-b"}
                    if output_series_discriminator == "engine"
                    else {"finished_reason": "variant-b"}
                )
                lines.extend(
                    (
                        _line(
                            "vllm:generation_tokens_total",
                            output_delta,
                            **first_extra,
                        ),
                        _line(
                            "vllm:generation_tokens_total",
                            0,
                            **second_extra,
                        ),
                    )
                )
        if "preemptions" not in omit:
            lines.append(
                _line(
                    "vllm:num_preemptions_total",
                    preemption_delta,
                    **scope("preemptions"),
                )
            )
        if "running" not in omit:
            lines.append(
                _line(
                    "vllm:num_requests_running",
                    running,
                    **scope("running"),
                )
            )
        if "waiting" not in omit:
            lines.append(
                _line(
                    "vllm:num_requests_waiting",
                    waiting,
                    **scope("waiting"),
                )
            )
        if swapped is not None and "swapped" not in omit:
            lines.append(
                _line(
                    "vllm:num_requests_swapped",
                    swapped,
                    **scope("swapped"),
                )
            )
        if "kv" not in omit:
            lines.append(
                _line(
                    "vllm:kv_cache_usage_perc",
                    kv_fraction,
                    **scope("kv"),
                )
            )
        histogram_means = {
            "ttft": mean_ttft_ms,
            "e2e": mean_e2e_ms,
            "queue": mean_queue_time_ms,
        }
        histogram_names = {
            "ttft": "vllm:time_to_first_token_seconds",
            "e2e": "vllm:e2e_request_latency_seconds",
            "queue": "vllm:request_queue_time_seconds",
        }
        for logical_name, mean_ms in histogram_means.items():
            if logical_name in omit:
                continue
            base = histogram_names[logical_name]
            observation_total = latency_observations.get(
                logical_name, requests_finished
            )
            observation_delta = observation_total * index // intervals
            lines.append(
                _line(
                    f"{base}_sum",
                    observation_delta * mean_ms / 1_000.0,
                    **scope(logical_name),
                )
            )
            lines.append(
                _line(
                    f"{base}_count",
                    observation_delta,
                    **scope(logical_name),
                )
            )
        snapshots.append(parse_vllm_metrics("\n".join(lines) + "\n"))
    return derive_metrics_window(
        snapshots[0],
        snapshots[-1],
        elapsed_seconds,
        observations=tuple(snapshots[1:-1]),
    )


def _context(**changes: object) -> BottleneckContext:
    values: dict[str, object] = {
        "current_max_num_seqs": 8,
        "current_max_num_batched_tokens": 32,
        "offered_concurrency": 16,
        "throttle_successful_requests": 40,
        "throttle_failed_requests": 0,
        "effective_flags_provenance": "runtime_verified",
        "traffic_scope": "operator_attested_exclusive",
    }
    values.update(changes)
    return BottleneckContext(**values)  # type: ignore[arg-type]


def _with_counts(
    window: MetricsWindow,
    *,
    remove: str | None = None,
    set_count: tuple[str, int] | None = None,
) -> MetricsWindow:
    counts = dict(window.series_counts)
    if remove is not None:
        counts.pop(remove)
    if set_count is not None:
        counts[set_count[0]] = set_count[1]
    return replace(window, series_counts=tuple(sorted(counts.items())))


def _with_elapsed(window: MetricsWindow, elapsed: float) -> MetricsWindow:
    return replace(
        window,
        elapsed_seconds=elapsed,
        finished_requests_per_second=(
            None
            if window.requests_finished is None
            else window.requests_finished / elapsed
        ),
        output_tokens_per_second=(
            None
            if window.output_tokens is None
            else window.output_tokens / elapsed
        ),
        prompt_tokens_per_second=(
            None
            if window.prompt_tokens is None
            else window.prompt_tokens / elapsed
        ),
    )


def _with_finished(window: MetricsWindow, count: int) -> MetricsWindow:
    latency_counts = dict(window.histogram_observation_counts)
    for name in ("ttft", "e2e", "queue"):
        if name in latency_counts:
            latency_counts[name] = count
    output_tokens = (
        None
        if window.output_tokens is None
        else max(window.output_tokens, count)
    )
    return replace(
        window,
        requests_finished=count,
        finished_requests_per_second=count / window.elapsed_seconds,
        output_tokens=output_tokens,
        output_tokens_per_second=(
            None
            if output_tokens is None
            else output_tokens / window.elapsed_seconds
        ),
        histogram_observation_counts=tuple(
            sorted(latency_counts.items())
        ),
        allowed_finished_requests=count,
        disallowed_finished_requests=0,
        unclassified_finished_requests=0,
    )


def _without_output(window: MetricsWindow) -> MetricsWindow:
    return replace(
        window,
        output_tokens=None,
        output_tokens_per_second=None,
    )


def _copy_as_subclass(value: object, subclass: type[object]) -> object:
    copied = object.__new__(subclass)
    for field in fields(value):  # type: ignore[arg-type]
        object.__setattr__(copied, field.name, getattr(value, field.name))
    return copied


class _EqualSecret(str):
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


class _ContextSubclass(BottleneckContext):
    pass


class _WindowSubclass(MetricsWindow):
    pass


class _SignalsSubclass(BottleneckSignals):
    pass


class _SuggestionSubclass(BottleneckSuggestion):
    pass


class _AnalysisSubclass(BottleneckAnalysis):
    pass


class BottleneckTestCase(unittest.TestCase):
    def assert_analysis_code(
        self,
        expected: str,
        callable_: object,
        *,
        secret: str = _PRIVATE_LABEL,
    ) -> None:
        with self.assertRaises(BottleneckAnalysisError) as raised:
            callable_()  # type: ignore[operator]
        self.assertEqual(raised.exception.code, expected)
        self.assertEqual(str(raised.exception), expected)
        self.assertEqual(raised.exception.args, (expected,))
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, repr(raised.exception))

    def assert_hard_safety_invariants(
        self, analysis: BottleneckAnalysis
    ) -> None:
        self.assertFalse(analysis.decision_eligible)
        self.assertEqual(analysis.decision_effect, "none")
        self.assertFalse(analysis.auto_apply)
        public = analysis.to_public_dict()
        self.assertIs(public["decision_eligible"], False)
        self.assertEqual(public["decision_effect"], "none")
        self.assertIs(public["auto_apply"], False)
        if analysis.suggestion is not None:
            suggestion = analysis.suggestion
            self.assertFalse(suggestion.auto_apply)
            self.assertFalse(suggestion.guaranteed_outcome)
            self.assertEqual(suggestion.decision_effect, "none")
            self.assertEqual(suggestion.claim_strength, "hypothesis_only")
            self.assertIn("suggestion", suggestion.label)
            self.assertIn("not_a_recommendation", suggestion.label)
            self.assertIn("may", suggestion.hypothesis.lower())
            self.assertIn("may", suggestion.risk.lower())
            self.assertNotIn("will improve", suggestion.hypothesis.lower())
            self.assertFalse(hasattr(suggestion, "apply"))


class FormulaAndThresholdTests(BottleneckTestCase):
    def test_increase_formula_has_explicit_ceil_math_and_bounds(self) -> None:
        cases = (
            (1, 8, 16, 1, 2),
            (4, 8, 16, 1, 5),
            (5, 20, 20, 2, 7),
            (8, 20, 32, 2, 10),
            (9, 30, 30, 3, 12),
            (8, 9, 32, 2, 9),
            (8, 32, 9, 2, 9),
        )
        base = _derived_window()
        for current, offered, batched, step, candidate in cases:
            successful = max(MIN_FINISHED_REQUESTS, 2 * offered)
            window = replace(
                _with_finished(base, successful),
                max_requests_running=current,
                max_requests_waiting=1,
            )
            context = _context(
                current_max_num_seqs=current,
                current_max_num_batched_tokens=batched,
                offered_concurrency=offered,
                throttle_successful_requests=successful,
            )
            with self.subTest(current=current, offered=offered, batched=batched):
                result = analyze_bottleneck(window, context)
                self.assertEqual(result.status, "suggestion_available")
                self.assertIsNotNone(result.suggestion)
                suggestion = result.suggestion
                assert suggestion is not None
                self.assertEqual(suggestion.direction, "increase")
                self.assertEqual(suggestion.policy_step, step)
                self.assertEqual(suggestion.candidate_max_num_seqs, candidate)
                self.assertEqual(
                    suggestion.formula_id,
                    "current_plus_ceil_25pct_bounded_by_verified_limits",
                )
                math_public = suggestion.to_public_dict()["math"]
                assert isinstance(math_public, dict)
                self.assertEqual(math_public["policy_step_numerator"], 1)
                self.assertEqual(math_public["policy_step_denominator"], 4)
                self.assertEqual(math_public["policy_step"], step)
                self.assert_hard_safety_invariants(result)

    def test_decrease_formula_has_explicit_ceil_math_and_floor(self) -> None:
        cases = (
            (2, 1, 1),
            (4, 1, 3),
            (5, 2, 3),
            (8, 2, 6),
            (9, 3, 6),
        )
        base = _derived_window(
            preemptions=1,
            kv_fraction=HIGH_KV_FRACTION,
        )
        for current, step, candidate in cases:
            offered = max(16, current)
            successful = max(MIN_FINISHED_REQUESTS, 2 * offered)
            window = replace(
                _with_finished(base, successful),
                max_requests_running=current,
                max_requests_waiting=0,
            )
            context = _context(
                current_max_num_seqs=current,
                current_max_num_batched_tokens=max(32, current),
                offered_concurrency=offered,
                throttle_successful_requests=successful,
            )
            with self.subTest(current=current):
                result = analyze_bottleneck(window, context)
                self.assertEqual(result.status, "suggestion_available")
                suggestion = result.suggestion
                assert suggestion is not None
                self.assertEqual(suggestion.direction, "decrease")
                self.assertEqual(suggestion.policy_step, step)
                self.assertEqual(suggestion.candidate_max_num_seqs, candidate)
                self.assertEqual(
                    suggestion.formula_id,
                    "current_minus_ceil_25pct_bounded_at_one",
                )
                self.assert_hard_safety_invariants(result)

    def test_signal_thresholds_are_inclusive_only_at_documented_edges(
        self,
    ) -> None:
        increase = analyze_bottleneck(
            _derived_window(
                kv_fraction=LOW_KV_FRACTION,
                mean_queue_time_ms=MATERIAL_QUEUE_SHARE * 1_000.0,
                mean_e2e_ms=1_000.0,
            ),
            _context(),
        )
        self.assertEqual(increase.status, "suggestion_available")
        assert increase.signals is not None
        self.assertEqual(
            increase.signals.max_kv_cache_usage_fraction,
            LOW_KV_FRACTION,
        )
        self.assertEqual(
            increase.signals.queue_share_of_mean_e2e,
            MATERIAL_QUEUE_SHARE,
        )

        below_queue = analyze_bottleneck(
            _derived_window(
                mean_queue_time_ms=math.nextafter(
                    MATERIAL_QUEUE_SHARE * 1_000.0, 0.0
                )
            ),
            _context(),
        )
        self.assertEqual(below_queue.status, "no_clear_signal")
        self.assertIn(
            "queue_share_below_policy_threshold",
            below_queue.no_suggestion_reasons,
        )

        above_low_kv = analyze_bottleneck(
            _derived_window(
                kv_fraction=math.nextafter(LOW_KV_FRACTION, 1.0)
            ),
            _context(),
        )
        self.assertEqual(above_low_kv.status, "no_clear_signal")
        self.assertIn(
            "queue_and_kv_signals_conflict",
            above_low_kv.no_suggestion_reasons,
        )

        decrease = analyze_bottleneck(
            _derived_window(
                preemptions=1,
                kv_fraction=HIGH_KV_FRACTION,
            ),
            _context(),
        )
        self.assertEqual(decrease.status, "suggestion_available")

        below_high_kv = analyze_bottleneck(
            _derived_window(
                preemptions=1,
                kv_fraction=math.nextafter(HIGH_KV_FRACTION, 0.0),
                waiting=0,
            ),
            _context(),
        )
        self.assertEqual(below_high_kv.status, "no_clear_signal")
        self.assertIn(
            "pressure_signal_incomplete",
            below_high_kv.no_suggestion_reasons,
        )

    def test_data_quality_thresholds_are_exact(self) -> None:
        exact_elapsed = analyze_bottleneck(
            _derived_window(elapsed_seconds=MIN_ELAPSED_SECONDS),
            _context(),
        )
        self.assertEqual(exact_elapsed.status, "suggestion_available")

        exact_spacing_elapsed = (
            MAX_AVERAGE_SAMPLE_SPACING_SECONDS
            * (MIN_OBSERVATIONS - 1)
        )
        exact_spacing = analyze_bottleneck(
            _derived_window(elapsed_seconds=exact_spacing_elapsed),
            _context(),
        )
        self.assertEqual(exact_spacing.status, "suggestion_available")
        assert exact_spacing.signals is not None
        self.assertEqual(
            exact_spacing.signals.average_sample_spacing_seconds,
            MAX_AVERAGE_SAMPLE_SPACING_SECONDS,
        )

        exact_finished = analyze_bottleneck(
            _derived_window(requests_finished=MIN_FINISHED_REQUESTS),
            _context(
                offered_concurrency=15,
                throttle_successful_requests=MIN_FINISHED_REQUESTS,
            ),
        )
        self.assertEqual(exact_finished.status, "suggestion_available")

        exact_concurrency_multiple = analyze_bottleneck(
            _derived_window(requests_finished=32),
            _context(throttle_successful_requests=32),
        )
        self.assertEqual(
            exact_concurrency_multiple.status, "suggestion_available"
        )

        just_below_concurrency_multiple = analyze_bottleneck(
            _derived_window(requests_finished=31),
            _context(throttle_successful_requests=31),
        )
        self.assertEqual(
            just_below_concurrency_multiple.status,
            "insufficient_evidence",
        )
        self.assertEqual(
            just_below_concurrency_multiple.quality_reasons,
            ("too_few_finished_requests",),
        )

    def test_golden_pair_marker_uses_the_generalized_reviewed_path(self) -> None:
        for pair in ((1, 8), (8, 1), (1, 2), (8, 6), (2, 8), (8, 10)):
            with self.subTest(pair=pair):
                self.assertEqual(
                    bottleneck_module._golden_support(*pair),
                    "supported_arbitrary_pair",
                )

        # Every candidate produced by the bounded 25% policy now has a
        # reviewed, counterbalanced validation path. This remains a marker;
        # the analyzer never runs Golden or upgrades its own eligibility.
        for current in range(1, 10_001):
            step = (current + 3) // 4
            increase = current + step
            decrease = max(1, current - step)
            self.assertEqual(
                bottleneck_module._golden_support(current, increase),
                "supported_arbitrary_pair",
            )
            if current != decrease:
                self.assertEqual(
                    bottleneck_module._golden_support(current, decrease),
                    "supported_arbitrary_pair",
                )

        actual = analyze_bottleneck(_derived_window(), _context())
        suggestion = actual.suggestion
        assert suggestion is not None
        self.assertEqual(
            suggestion.current_golden_support,
            "supported_arbitrary_pair",
        )
        self.assertEqual(
            suggestion.required_next_step,
            "run_generalized_golden_protocol_at_offered_concurrency",
        )
        object.__setattr__(
            suggestion, "current_golden_support", "unsupported_candidate_pair"
        )
        object.__setattr__(
            suggestion,
            "required_next_step",
            "no_decision_path_until_counterbalanced_protocol_supports_pair",
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output", actual.to_public_dict
        )


class EvidenceAndNoSignalTests(BottleneckTestCase):
    def test_every_quality_reason_is_fail_closed_and_suppresses_signals(
        self,
    ) -> None:
        base_window = _derived_window()
        base_context = _context()
        without_output_source = tuple(
            name
            for name in base_window.source_families
            if name != "vllm:generation_tokens_total"
        )
        cases: tuple[
            tuple[str | tuple[str, ...], MetricsWindow, BottleneckContext],
            ...,
        ] = (
            (
                "effective_flags_not_runtime_verified",
                base_window,
                _context(effective_flags_provenance="operator_attested"),
            ),
            (
                "exporter_request_count_mismatch",
                base_window,
                _context(throttle_successful_requests=39),
            ),
            (
                "finish_reason_evidence_not_clean",
                replace(
                    base_window,
                    allowed_finished_requests=39,
                    disallowed_finished_requests=1,
                    unclassified_finished_requests=0,
                ),
                base_context,
            ),
            (
                "gauge_exceeds_offered_concurrency",
                replace(base_window, max_requests_waiting=17),
                base_context,
            ),
            (
                "latency_metrics_inconsistent",
                replace(base_window, mean_queue_time_ms=301.0),
                base_context,
            ),
            (
                "latency_observation_count_mismatch",
                replace(
                    base_window,
                    histogram_observation_counts=(
                        ("e2e", 39),
                        ("queue", 40),
                        ("ttft", 40),
                    ),
                ),
                base_context,
            ),
            (
                "max_num_batched_tokens_below_max_num_seqs",
                base_window,
                _context(current_max_num_batched_tokens=7),
            ),
            (
                "metric_scope_not_single_consistent",
                replace(
                    base_window,
                    metric_scope_consistent=False,
                    metric_scope_count=2,
                ),
                base_context,
            ),
            (
                "missing_required_metric",
                _without_output(base_window),
                base_context,
            ),
            (
                "counter_rate_inconsistent",
                replace(
                    base_window,
                    finished_requests_per_second=5.5,
                ),
                base_context,
            ),
            (
                (
                    "nonpositive_output_evidence",
                    "output_token_count_below_finished_requests",
                ),
                replace(
                    base_window,
                    output_tokens=0,
                    output_tokens_per_second=0.0,
                ),
                base_context,
            ),
            (
                "output_token_count_below_finished_requests",
                replace(
                    base_window,
                    output_tokens=39,
                    output_tokens_per_second=39 / 8.0,
                ),
                base_context,
            ),
            (
                "required_metric_series_count_not_one",
                _with_counts(base_window, remove="output_tokens"),
                base_context,
            ),
            (
                "required_metric_source_missing",
                replace(
                    base_window,
                    source_families=without_output_source,
                ),
                base_context,
            ),
            (
                "running_exceeds_current_max_num_seqs",
                replace(base_window, max_requests_running=9),
                base_context,
            ),
            (
                "sample_spacing_too_sparse",
                _with_elapsed(
                    base_window,
                    math.nextafter(20.0, math.inf),
                ),
                base_context,
            ),
            (
                "throttle_requests_failed",
                base_window,
                _context(throttle_failed_requests=1),
            ),
            (
                "too_few_finished_requests",
                _with_finished(base_window, 29),
                _context(throttle_successful_requests=29),
            ),
            (
                "too_few_observations",
                replace(base_window, observations=MIN_OBSERVATIONS - 1),
                base_context,
            ),
            (
                "traffic_scope_not_exclusive",
                base_window,
                _context(traffic_scope="unconfirmed"),
            ),
            (
                "window_too_short",
                _with_elapsed(
                    base_window,
                    math.nextafter(
                        MIN_ELAPSED_SECONDS, 0.0
                    ),
                ),
                base_context,
            ),
        )
        observed: set[str] = set()
        for expected, window, context in cases:
            expected_reasons = (
                expected if isinstance(expected, tuple) else (expected,)
            )
            with self.subTest(expected=expected_reasons):
                result = analyze_bottleneck(window, context)
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertEqual(
                    result.quality_reasons,
                    tuple(sorted(expected_reasons)),
                )
                self.assertEqual(result.no_suggestion_reasons, ())
                self.assertIsNone(result.signals)
                self.assertIsNone(result.suggestion)
                self.assert_hard_safety_invariants(result)
                observed.update(result.quality_reasons)
        self.assertEqual(observed, _QUALITY_REASONS)

    def test_real_missing_metrics_report_all_evidence_gaps(self) -> None:
        window = _derived_window(omit=frozenset({"output"}))
        self.assertIsNone(window.output_tokens)
        self.assertNotIn(
            "vllm:generation_tokens_total", window.source_families
        )
        self.assertNotIn("output_tokens", dict(window.series_counts))
        result = analyze_bottleneck(window, _context())
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            set(result.quality_reasons),
            {
                "missing_required_metric",
                "required_metric_series_count_not_one",
                "required_metric_source_missing",
            },
        )

    def test_real_multiseries_required_metric_is_rejected(self) -> None:
        window = _derived_window(output_series=2)
        self.assertEqual(dict(window.series_counts)["output_tokens"], 2)
        result = analyze_bottleneck(window, _context())
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            set(result.quality_reasons),
            {
                "metric_scope_not_single_consistent",
                "required_metric_series_count_not_one",
            },
        )

    def test_finished_reason_series_share_one_metric_scope(self) -> None:
        window = _derived_window(request_series=2)
        self.assertEqual(dict(window.series_counts)["requests_finished"], 2)
        self.assertTrue(window.metric_scope_consistent)
        self.assertEqual(window.metric_scope_count, 1)
        self.assertEqual(window.allowed_finished_requests, 40)
        self.assertEqual(window.disallowed_finished_requests, 0)
        self.assertEqual(window.unclassified_finished_requests, 0)
        result = analyze_bottleneck(window, _context())
        self.assertEqual(result.status, "suggestion_available")

    def test_allowed_finish_reason_evidence_is_accepted(self) -> None:
        for reasons in (("stop",), ("length",), ("stop", "length")):
            with self.subTest(reasons=reasons):
                window = _derived_window(request_finish_reasons=reasons)
                self.assertEqual(window.requests_finished, 40)
                self.assertEqual(window.allowed_finished_requests, 40)
                self.assertEqual(window.disallowed_finished_requests, 0)
                self.assertEqual(window.unclassified_finished_requests, 0)
                result = analyze_bottleneck(window, _context())
                self.assertEqual(result.status, "suggestion_available")
                self.assertEqual(result.quality_reasons, ())

    def test_unclean_finish_reason_evidence_is_fixed_and_label_free(
        self,
    ) -> None:
        unknown = "private-unknown-finish-payload"
        cases = (
            ("abort", 20, 20, 0),
            ("error", 20, 20, 0),
            ("repetition", 20, 20, 0),
            (unknown, 20, 0, 20),
            (None, 20, 0, 20),
        )
        for reason, allowed, disallowed, unclassified in cases:
            with self.subTest(reason_class=reason):
                window = _derived_window(
                    request_finish_reasons=("length", reason)
                )
                self.assertEqual(window.requests_finished, 40)
                self.assertEqual(
                    window.allowed_finished_requests, allowed
                )
                self.assertEqual(
                    window.disallowed_finished_requests, disallowed
                )
                self.assertEqual(
                    window.unclassified_finished_requests, unclassified
                )
                result = analyze_bottleneck(window, _context())
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertEqual(
                    result.quality_reasons,
                    ("finish_reason_evidence_not_clean",),
                )
                self.assertIsNone(result.signals)
                self.assertIsNone(result.suggestion)
                rendered = json.dumps(
                    {
                        "window": window.to_public_dict(),
                        "analysis": result.to_public_dict(),
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                rendered += repr(window) + repr(result)
                self.assertNotIn(_PRIVATE_LABEL, rendered)
                self.assertNotIn(unknown, rendered)
                self.assertIsNone(
                    re.search(r"\b[0-9a-f]{64}\b", rendered)
                )

    def test_finished_reason_is_scope_only_for_request_success(self) -> None:
        window = _derived_window(
            output_series=2,
            output_series_discriminator="finished_reason",
        )
        self.assertEqual(dict(window.series_counts)["output_tokens"], 2)
        self.assertFalse(window.metric_scope_consistent)
        self.assertGreaterEqual(window.metric_scope_count or 0, 2)
        result = analyze_bottleneck(window, _context())
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            set(result.quality_reasons),
            {
                "metric_scope_not_single_consistent",
                "required_metric_series_count_not_one",
            },
        )

    def test_real_latency_observation_count_mismatches_fail_closed(
        self,
    ) -> None:
        cases = (
            {"ttft": 1, "e2e": 1, "queue": 1},
            {"ttft": 40, "e2e": 39, "queue": 38},
        )
        for counts in cases:
            with self.subTest(counts=counts):
                window = _derived_window(
                    latency_observations=counts
                )
                self.assertEqual(
                    dict(window.histogram_observation_counts), counts
                )
                self.assertEqual(window.requests_finished, 40)
                result = analyze_bottleneck(window, _context())
                self.assertEqual(result.status, "insufficient_evidence")
                self.assertEqual(
                    result.quality_reasons,
                    ("latency_observation_count_mismatch",),
                )
                self.assertIsNone(result.signals)
                self.assertIsNone(result.suggestion)

    def test_real_cross_family_metric_scope_mismatch_is_label_free(
        self,
    ) -> None:
        window = _derived_window(scope_mismatch_family="output")
        self.assertFalse(window.metric_scope_consistent)
        self.assertEqual(window.metric_scope_count, 2)
        self.assertTrue(
            all(count == 1 for _, count in window.series_counts)
        )
        result = analyze_bottleneck(window, _context())
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.quality_reasons,
            ("metric_scope_not_single_consistent",),
        )
        rendered = json.dumps(
            {
                "window": window.to_public_dict(),
                "analysis": result.to_public_dict(),
            },
            sort_keys=True,
            allow_nan=False,
        )
        rendered += repr(window) + repr(result)
        self.assertNotIn(_PRIVATE_LABEL, rendered)
        self.assertNotIn(_SECOND_PRIVATE_LABEL, rendered)
        self.assertNotIn("private-worker-z", rendered)
        self.assertIsNone(re.search(r"\b[0-9a-f]{64}\b", rendered))

    def test_alias_source_ambiguity_is_rejected(self) -> None:
        window = _derived_window()
        ambiguous_sources = tuple(
            sorted(
                (*window.source_families, "vllm:gpu_cache_usage_perc")
            )
        )
        ambiguous = replace(window, source_families=ambiguous_sources)
        result = analyze_bottleneck(ambiguous, _context())
        self.assertEqual(
            result.quality_reasons,
            ("required_metric_source_missing",),
        )

    def test_legacy_metrics_window_positional_tail_remains_compatible(
        self,
    ) -> None:
        current = _derived_window()
        legacy_positional_values = (
            current.elapsed_seconds,
            current.observations,
            current.requests_finished,
            current.output_tokens,
            current.prompt_tokens,
            current.preemptions,
            current.finished_requests_per_second,
            current.output_tokens_per_second,
            current.prompt_tokens_per_second,
            current.mean_ttft_ms,
            current.mean_tpot_ms,
            current.mean_inter_token_latency_ms,
            current.mean_e2e_ms,
            current.mean_queue_time_ms,
            current.mean_prefill_time_ms,
            current.mean_decode_time_ms,
            current.max_requests_running,
            current.max_requests_waiting,
            current.max_requests_swapped,
            current.max_kv_cache_usage_fraction,
            current.source_families,
            current.series_counts,
            current.scope,
            current.decision_effect,
            current.caveats,
        )
        legacy = MetricsWindow(*legacy_positional_values)
        self.assertEqual(legacy.scope, current.scope)
        self.assertEqual(legacy.decision_effect, current.decision_effect)
        self.assertEqual(legacy.caveats, current.caveats)
        self.assertEqual(legacy.histogram_observation_counts, ())
        self.assertIsNone(legacy.metric_scope_consistent)
        self.assertIsNone(legacy.metric_scope_count)
        self.assertIsNone(legacy.allowed_finished_requests)
        self.assertIsNone(legacy.disallowed_finished_requests)
        self.assertIsNone(legacy.unclassified_finished_requests)
        json.dumps(legacy.to_public_dict(), allow_nan=False)

        result = analyze_bottleneck(legacy, _context())
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            set(result.quality_reasons),
            {
                "finish_reason_evidence_not_clean",
                "latency_observation_count_mismatch",
                "metric_scope_not_single_consistent",
            },
        )
        self.assertIsNone(result.signals)
        self.assertIsNone(result.suggestion)

    def test_every_no_suggestion_reason_is_explicit(self) -> None:
        base = _derived_window()
        pressure = replace(
            base,
            preemptions=1,
            max_kv_cache_usage_fraction=HIGH_KV_FRACTION,
            max_requests_waiting=0,
        )
        cases: tuple[
            tuple[str, MetricsWindow, BottleneckContext, set[str]], ...
        ] = (
            (
                "configured_limit_not_exercised",
                replace(base, max_requests_running=7),
                _context(),
                {"configured_limit_not_exercised"},
            ),
            (
                "legacy_swap_pressure_observed",
                replace(base, max_requests_swapped=1),
                _context(),
                {"legacy_swap_pressure_observed"},
            ),
            (
                "no_higher_candidate_within_verified_bounds",
                base,
                _context(current_max_num_batched_tokens=8),
                {"no_higher_candidate_within_verified_bounds"},
            ),
            (
                "no_lower_candidate",
                replace(pressure, max_requests_running=1),
                _context(current_max_num_seqs=1),
                {"no_lower_candidate"},
            ),
            (
                "no_material_queue",
                replace(base, max_requests_waiting=0),
                _context(),
                {"no_material_queue"},
            ),
            (
                "pressure_signal_incomplete",
                replace(
                    base,
                    preemptions=1,
                    max_kv_cache_usage_fraction=LOW_KV_FRACTION,
                ),
                _context(),
                {"pressure_signal_incomplete"},
            ),
            (
                "queue_share_below_policy_threshold",
                replace(base, mean_queue_time_ms=199.0),
                _context(),
                {"queue_share_below_policy_threshold"},
            ),
            (
                "queue_and_kv_signals_conflict",
                replace(base, max_kv_cache_usage_fraction=0.8),
                _context(),
                {
                    "kv_headroom_not_clear",
                    "queue_and_kv_signals_conflict",
                },
            ),
            (
                "kv_headroom_not_clear",
                replace(base, max_kv_cache_usage_fraction=0.8),
                _context(),
                {
                    "kv_headroom_not_clear",
                    "queue_and_kv_signals_conflict",
                },
            ),
        )
        observed: set[str] = set()
        for name, window, context, expected in cases:
            with self.subTest(name=name):
                result = analyze_bottleneck(window, context)
                self.assertEqual(result.status, "no_clear_signal")
                self.assertEqual(set(result.no_suggestion_reasons), expected)
                self.assertEqual(result.quality_reasons, ())
                self.assertIsNotNone(result.signals)
                self.assertIsNone(result.suggestion)
                self.assert_hard_safety_invariants(result)
                observed.update(result.no_suggestion_reasons)
        self.assertEqual(observed, _NO_SUGGESTION_REASONS)

    def test_multiple_quality_failures_are_sorted_and_never_guessed_past(
        self,
    ) -> None:
        result = analyze_bottleneck(
            replace(
                _with_finished(
                    _with_elapsed(_derived_window(), 4.0), 1
                ),
                observations=4,
                output_tokens=None,
                output_tokens_per_second=None,
            ),
            _context(
                throttle_successful_requests=2,
                throttle_failed_requests=1,
                effective_flags_provenance="unknown",
                traffic_scope="unconfirmed",
            ),
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.quality_reasons,
            tuple(sorted(set(result.quality_reasons))),
        )
        self.assertGreater(len(result.quality_reasons), 5)
        self.assertIsNone(result.signals)
        self.assertIsNone(result.suggestion)


class ValidationAndSerializationTests(BottleneckTestCase):
    def test_signal_math_and_evidence_inputs_are_explicit(self) -> None:
        result = analyze_bottleneck(_derived_window(), _context())
        signals = result.signals
        self.assertIsNotNone(signals)
        assert signals is not None
        self.assertEqual(signals.elapsed_seconds, 8.0)
        self.assertEqual(signals.observations, 5)
        self.assertEqual(signals.average_sample_spacing_seconds, 2.0)
        self.assertAlmostEqual(signals.sampled_kv_headroom_fraction, 0.3)
        self.assertEqual(signals.queue_share_of_mean_e2e, 0.2)
        self.assertEqual(signals.preemptions_per_100_finished, 0.0)
        self.assertEqual(signals.finished_requests, 40)
        self.assertEqual(signals.finished_requests_per_second, 5.0)
        self.assertEqual(signals.latency_observations, 40)
        self.assertEqual(signals.max_requests_running, 8)
        self.assertEqual(signals.max_requests_waiting, 4)
        self.assertEqual(signals.max_requests_swapped, 0)
        self.assertEqual(signals.current_max_num_seqs, 8)
        self.assertEqual(signals.current_max_num_batched_tokens, 32)
        self.assertEqual(signals.offered_concurrency, 16)
        self.assertTrue(signals.configured_limit_exercised)
        self.assertEqual(signals.output_tokens, 400)
        self.assertEqual(signals.output_tokens_per_second, 50.0)
        self.assertEqual(signals.allowed_finished_requests, 40)
        self.assertEqual(signals.disallowed_finished_requests, 0)
        self.assertEqual(signals.unclassified_finished_requests, 0)
        self.assertEqual(
            signals.to_public_dict(),
            result.to_public_dict()["signals"],
        )

        pressure = analyze_bottleneck(
            _derived_window(
                preemptions=2,
                kv_fraction=HIGH_KV_FRACTION,
            ),
            _context(),
        )
        assert pressure.signals is not None
        self.assertEqual(
            pressure.signals.preemptions_per_100_finished, 5.0
        )

    def test_tampered_signal_derivations_are_rejected(self) -> None:
        changes = (
            ("elapsed_seconds", 9.0),
            ("observations", 6),
            ("observations", 1),
            ("observations", MAX_OBSERVATIONS + 3),
            ("average_sample_spacing_seconds", 2.1),
            ("sampled_kv_headroom_fraction", 0.2),
            ("queue_share_of_mean_e2e", 0.3),
            ("mean_e2e_ms", 0.0),
            ("mean_ttft_ms", 100.0),
            ("mean_ttft_ms", 1_100.0),
            ("preemptions_per_100_finished", 1.0),
            ("output_tokens", 401),
            ("finished_requests", 39),
            ("finished_requests_per_second", 4.9),
            ("allowed_finished_requests", 39),
            ("disallowed_finished_requests", 1),
            ("unclassified_finished_requests", 1),
            ("latency_observations", 39),
            ("current_max_num_seqs", 9),
            ("current_max_num_batched_tokens", 7),
            ("offered_concurrency", 7),
            ("configured_limit_exercised", False),
        )
        for name, value in changes:
            result = analyze_bottleneck(_derived_window(), _context())
            assert result.signals is not None
            object.__setattr__(result.signals, name, value)
            with self.subTest(field=name):
                self.assert_analysis_code(
                    "bottleneck_invalid_output", result.to_public_dict
                )

    def test_exact_signal_thresholds_cannot_be_crossed_by_forgery(
        self,
    ) -> None:
        below_queue = analyze_bottleneck(
            replace(
                _derived_window(),
                mean_queue_time_ms=math.nextafter(200.0, 0.0),
            ),
            _context(),
        )
        self.assertEqual(below_queue.status, "no_clear_signal")
        assert below_queue.signals is not None
        self.assertLess(
            below_queue.signals.queue_share_of_mean_e2e,
            MATERIAL_QUEUE_SHARE,
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            lambda: replace(
                below_queue.signals,
                queue_share_of_mean_e2e=MATERIAL_QUEUE_SHARE,
            ),
        )

        increase = analyze_bottleneck(_derived_window(), _context())
        assert increase.suggestion is not None
        object.__setattr__(
            below_queue.signals,
            "queue_share_of_mean_e2e",
            MATERIAL_QUEUE_SHARE,
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            lambda: BottleneckAnalysis(
                status="suggestion_available",
                quality_reasons=(),
                no_suggestion_reasons=(),
                signals=below_queue.signals,
                suggestion=increase.suggestion,
            ),
        )

        exact_spacing = analyze_bottleneck(
            _derived_window(elapsed_seconds=20.0), _context()
        )
        assert exact_spacing.signals is not None
        overlong = math.nextafter(20.0, math.inf)
        common_changes = {
            "elapsed_seconds": overlong,
            "finished_requests_per_second": 40 / overlong,
            "output_tokens_per_second": 400 / overlong,
        }
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            lambda: replace(
                exact_spacing.signals,
                average_sample_spacing_seconds=5.0,
                **common_changes,
            ),
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            lambda: replace(
                exact_spacing.signals,
                average_sample_spacing_seconds=overlong / 4,
                **common_changes,
            ),
        )

    def test_signal_latency_order_is_enforced_at_construction(self) -> None:
        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        for mean_ttft_ms in (100.0, 1_100.0):
            with self.subTest(mean_ttft_ms=mean_ttft_ms):
                self.assert_analysis_code(
                    "bottleneck_invalid_output",
                    lambda mean_ttft_ms=mean_ttft_ms: replace(
                        result.signals,
                        mean_ttft_ms=mean_ttft_ms,
                    ),
                )

    def test_derived_float_tolerance_cannot_cross_policy_gates(self) -> None:
        below = math.nextafter(MATERIAL_QUEUE_SHARE, 0.0)
        no_clear = analyze_bottleneck(
            replace(
                _derived_window(),
                mean_queue_time_ms=below * 1_000.0,
            ),
            _context(),
        )
        increase = analyze_bottleneck(_derived_window(), _context())
        assert no_clear.signals is not None
        assert increase.suggestion is not None
        object.__setattr__(
            no_clear.signals,
            "queue_share_of_mean_e2e",
            MATERIAL_QUEUE_SHARE,
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            lambda: BottleneckAnalysis(
                status="suggestion_available",
                quality_reasons=(),
                no_suggestion_reasons=(),
                signals=no_clear.signals,
                suggestion=increase.suggestion,
            ),
        )

        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        object.__setattr__(
            result.signals,
            "elapsed_seconds",
            math.nextafter(20.0, math.inf),
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output", result.to_public_dict
        )

    def test_public_projection_is_finite_label_free_json(self) -> None:
        results = (
            analyze_bottleneck(_derived_window(), _context()),
            analyze_bottleneck(
                _derived_window(
                    preemptions=1,
                    kv_fraction=HIGH_KV_FRACTION,
                ),
                _context(),
            ),
            analyze_bottleneck(
                replace(_derived_window(), max_requests_waiting=0),
                _context(),
            ),
            analyze_bottleneck(
                _derived_window(),
                _context(traffic_scope="unconfirmed"),
            ),
        )
        for result in results:
            with self.subTest(status=result.status):
                public = result.to_public_dict()
                encoded = json.dumps(
                    public,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self.assertNotIn(_PRIVATE_LABEL, encoded)
                self.assertNotIn("api-key", encoded)
                self.assertEqual(
                    public["schema_version"], ANALYSIS_SCHEMA_VERSION
                )
                self.assertEqual(
                    public["artifact_type"], ANALYSIS_ARTIFACT_TYPE
                )
                self.assert_hard_safety_invariants(result)

    def test_public_projection_returns_fresh_containers(self) -> None:
        result = analyze_bottleneck(_derived_window(), _context())
        first = result.to_public_dict()
        first["quality_reasons"] = ["forged"]
        caveats = first["caveats"]
        assert isinstance(caveats, list)
        caveats.clear()
        suggestion = first["suggestion"]
        assert isinstance(suggestion, dict)
        suggestion["auto_apply"] = True
        math_values = suggestion["math"]
        assert isinstance(math_values, dict)
        math_values["policy_step"] = -1

        second = result.to_public_dict()
        self.assertEqual(second["quality_reasons"], [])
        self.assertTrue(second["caveats"])
        second_suggestion = second["suggestion"]
        assert isinstance(second_suggestion, dict)
        self.assertIs(second_suggestion["auto_apply"], False)
        second_math = second_suggestion["math"]
        assert isinstance(second_math, dict)
        self.assertGreater(second_math["policy_step"], 0)

    def test_inputs_are_unchanged_and_analysis_performs_no_io(self) -> None:
        window = _derived_window()
        context = _context()
        window_before = json.dumps(
            window.to_public_dict(), sort_keys=True, allow_nan=False
        )
        context_before = tuple(
            getattr(context, field.name) for field in fields(context)
        )
        source_identity = window.source_families
        counts_identity = window.series_counts

        with (
            patch("builtins.open") as open_mock,
            patch.object(socket, "create_connection") as connect_mock,
            patch.object(subprocess, "Popen") as popen_mock,
        ):
            result = analyze_bottleneck(window, context)
        open_mock.assert_not_called()
        connect_mock.assert_not_called()
        popen_mock.assert_not_called()
        self.assertEqual(
            json.dumps(
                window.to_public_dict(), sort_keys=True, allow_nan=False
            ),
            window_before,
        )
        self.assertEqual(
            tuple(getattr(context, field.name) for field in fields(context)),
            context_before,
        )
        self.assertIs(window.source_families, source_identity)
        self.assertIs(window.series_counts, counts_identity)
        self.assertEqual(result.status, "suggestion_available")

    def test_frozen_dataclasses_reject_normal_mutation(self) -> None:
        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        assert result.suggestion is not None
        objects_and_fields = (
            (_context(), "current_max_num_seqs", 99),
            (result.signals, "preemptions", 99),
            (result.suggestion, "auto_apply", True),
            (result, "decision_eligible", True),
        )
        for value, name, replacement in objects_and_fields:
            with self.subTest(type=type(value).__name__, field=name):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, name, replacement)

    def test_replace_rejects_invalid_context_and_output_invariants(self) -> None:
        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        assert result.suggestion is not None
        cases = (
            (
                "bottleneck_invalid_context",
                lambda: replace(_context(), current_max_num_seqs=True),
            ),
            (
                "bottleneck_invalid_output",
                lambda: replace(result.signals, mean_e2e_ms=math.nan),
            ),
            (
                "bottleneck_invalid_output",
                lambda: replace(result.suggestion, auto_apply=True),
            ),
            (
                "bottleneck_invalid_output",
                lambda: replace(result.suggestion, guaranteed_outcome=True),
            ),
            (
                "bottleneck_invalid_output",
                lambda: replace(
                    result.suggestion,
                    label="guaranteed_configuration_recommendation",
                ),
            ),
            (
                "bottleneck_invalid_output",
                lambda: replace(result, decision_eligible=True),
            ),
            (
                "bottleneck_invalid_output",
                lambda: replace(result, auto_apply=True),
            ),
        )
        for expected, callable_ in cases:
            with self.subTest(expected=expected):
                self.assert_analysis_code(expected, callable_)

    def test_object_level_tampering_is_detected_before_projection(self) -> None:
        private = "https://private.example/token/super-secret"
        context = _context()
        object.__setattr__(context, "traffic_scope", _EqualSecret(private))
        self.assert_analysis_code(
            "bottleneck_invalid_context",
            lambda: analyze_bottleneck(_derived_window(), context),
            secret=private,
        )

        window = _derived_window()
        object.__setattr__(window, "scope", private)
        self.assert_analysis_code(
            "bottleneck_invalid_metrics_window",
            lambda: analyze_bottleneck(window, _context()),
            secret=private,
        )

        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        object.__setattr__(
            result.signals, "output_tokens_per_second", math.inf
        )
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            result.to_public_dict,
            secret=private,
        )

        result = analyze_bottleneck(_derived_window(), _context())
        assert result.suggestion is not None
        object.__setattr__(result.suggestion, "label", private)
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            result.to_public_dict,
            secret=private,
        )

        result = analyze_bottleneck(_derived_window(), _context())
        object.__setattr__(result, "status", private)
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            result.to_public_dict,
            secret=private,
        )

    def test_method_shadowing_cannot_bypass_base_validators(self) -> None:
        private = "private-shadowed-method-payload"

        window = _derived_window()
        object.__setattr__(
            window,
            "to_public_dict",
            lambda: {"scope": "forged", "secret": private},
        )
        valid = analyze_bottleneck(window, _context())
        self.assertEqual(valid.status, "suggestion_available")
        self.assertNotIn(private, json.dumps(valid.to_public_dict()))

        invalid_window = _derived_window()
        object.__setattr__(
            invalid_window,
            "to_public_dict",
            lambda: {"scope": "server_exporter_window"},
        )
        object.__setattr__(invalid_window, "scope", private)
        self.assert_analysis_code(
            "bottleneck_invalid_metrics_window",
            lambda: analyze_bottleneck(invalid_window, _context()),
            secret=private,
        )

        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        assert result.suggestion is not None
        forged_signals = _copy_as_subclass(
            result.signals, _SignalsSubclass
        )
        forged_suggestion = _copy_as_subclass(
            result.suggestion, _SuggestionSubclass
        )
        object.__setattr__(
            forged_signals,
            "to_public_dict",
            lambda: {"private": private},
        )
        object.__setattr__(
            forged_suggestion,
            "to_public_dict",
            lambda: {
                "auto_apply": True,
                "guaranteed_outcome": True,
                "private": private,
            },
        )
        object.__setattr__(result, "signals", forged_signals)
        object.__setattr__(result, "suggestion", forged_suggestion)
        self.assert_analysis_code(
            "bottleneck_invalid_output",
            result.to_public_dict,
            secret=private,
        )

    def test_direct_analysis_rejects_contradictory_suggestions_and_reasons(
        self,
    ) -> None:
        increase = analyze_bottleneck(_derived_window(), _context())
        pressure = analyze_bottleneck(
            _derived_window(
                preemptions=1,
                kv_fraction=HIGH_KV_FRACTION,
                waiting=0,
            ),
            _context(),
        )
        not_exercised = analyze_bottleneck(
            replace(_derived_window(), max_requests_running=7),
            _context(),
        )
        below_queue = analyze_bottleneck(
            replace(_derived_window(), mean_queue_time_ms=199.0),
            _context(),
        )
        kv_conflict = analyze_bottleneck(
            replace(
                _derived_window(), max_kv_cache_usage_fraction=0.8
            ),
            _context(),
        )
        swapped = analyze_bottleneck(
            replace(_derived_window(), max_requests_swapped=1),
            _context(),
        )
        other_context = _context(
            current_max_num_seqs=9,
            current_max_num_batched_tokens=40,
            offered_concurrency=30,
            throttle_successful_requests=60,
        )
        other_window = replace(
            _with_finished(_derived_window(), 60),
            max_requests_running=9,
        )
        other = analyze_bottleneck(other_window, other_context)

        assert increase.signals is not None
        assert increase.suggestion is not None
        assert pressure.signals is not None
        assert pressure.suggestion is not None
        assert not_exercised.signals is not None
        assert below_queue.signals is not None
        assert kv_conflict.signals is not None
        assert swapped.signals is not None
        assert other.suggestion is not None

        contradictory_pairs = (
            (not_exercised.signals, increase.suggestion),
            (below_queue.signals, increase.suggestion),
            (kv_conflict.signals, increase.suggestion),
            (pressure.signals, increase.suggestion),
            (increase.signals, pressure.suggestion),
            (increase.signals, other.suggestion),
        )
        for signals, suggestion in contradictory_pairs:
            with self.subTest(
                signal_preemptions=signals.preemptions,
                suggestion_direction=suggestion.direction,
                suggestion_current=suggestion.current_max_num_seqs,
            ):
                self.assert_analysis_code(
                    "bottleneck_invalid_output",
                    lambda signals=signals, suggestion=suggestion: (
                        BottleneckAnalysis(
                            status="suggestion_available",
                            quality_reasons=(),
                            no_suggestion_reasons=(),
                            signals=signals,
                            suggestion=suggestion,
                        )
                    ),
                )

        self.assert_analysis_code(
            "bottleneck_invalid_output",
            lambda: BottleneckAnalysis(
                status="no_clear_signal",
                quality_reasons=(),
                no_suggestion_reasons=("no_material_queue",),
                signals=swapped.signals,
                suggestion=None,
            ),
        )

    def test_exact_type_checks_reject_dataclass_subclasses(self) -> None:
        result = analyze_bottleneck(_derived_window(), _context())
        assert result.signals is not None
        assert result.suggestion is not None
        forged_context = _copy_as_subclass(
            _context(), _ContextSubclass
        )
        forged_window = _copy_as_subclass(
            _derived_window(), _WindowSubclass
        )
        forged_signals = _copy_as_subclass(
            result.signals, _SignalsSubclass
        )
        forged_suggestion = _copy_as_subclass(
            result.suggestion, _SuggestionSubclass
        )
        forged_analysis = _copy_as_subclass(result, _AnalysisSubclass)
        cases = (
            (
                "bottleneck_invalid_context",
                lambda: analyze_bottleneck(
                    _derived_window(), forged_context  # type: ignore[arg-type]
                ),
            ),
            (
                "bottleneck_invalid_metrics_window",
                lambda: analyze_bottleneck(
                    forged_window, _context()  # type: ignore[arg-type]
                ),
            ),
            (
                "bottleneck_invalid_output",
                forged_signals.to_public_dict,  # type: ignore[attr-defined]
            ),
            (
                "bottleneck_invalid_output",
                forged_suggestion.to_public_dict,  # type: ignore[attr-defined]
            ),
            (
                "bottleneck_invalid_output",
                forged_analysis.to_public_dict,  # type: ignore[attr-defined]
            ),
        )
        for expected, callable_ in cases:
            with self.subTest(expected=expected):
                self.assert_analysis_code(expected, callable_)

    def test_arbitrary_inputs_fail_with_fixed_nonreflective_codes(self) -> None:
        private = "private://credentials/should-never-reflect"
        invalid_contexts = (
            private,
            _EqualSecret(private),
            {"current_max_num_seqs": private},
            [private],
            object(),
        )
        for value in invalid_contexts:
            with self.subTest(context_type=type(value).__name__):
                self.assert_analysis_code(
                    "bottleneck_invalid_context",
                    lambda value=value: analyze_bottleneck(
                        _derived_window(), value  # type: ignore[arg-type]
                    ),
                    secret=private,
                )
        invalid_windows = (
            private,
            _EqualSecret(private),
            {"source_families": [private]},
            [private],
            object(),
        )
        for value in invalid_windows:
            with self.subTest(window_type=type(value).__name__):
                self.assert_analysis_code(
                    "bottleneck_invalid_metrics_window",
                    lambda value=value: analyze_bottleneck(
                        value, _context()  # type: ignore[arg-type]
                    ),
                    secret=private,
                )


class DeterministicPropertyTests(BottleneckTestCase):
    def test_sixty_thousand_numeric_and_state_combinations_fail_closed(
        self,
    ) -> None:
        seed = 0x600DF00D
        total_cases = 60_000
        rng = random.Random(seed)
        increase_base = _derived_window()
        pressure_base = _derived_window(
            preemptions=1,
            kv_fraction=HIGH_KV_FRACTION,
            waiting=0,
        )
        artifact_counts = {
            "insufficient_evidence": 0,
            "no_clear_signal": 0,
            "suggestion_available": 0,
        }
        error_count = 0
        fixed_error_codes = {
            "bottleneck_invalid_context",
            "bottleneck_invalid_metrics_window",
            "bottleneck_invalid_output",
        }

        for index in range(total_cases):
            mode = index % 8
            if mode in {0, 1, 2}:
                current = rng.randint(1 if mode != 1 else 2, 1_024)
                offered = current + rng.randint(1, 512)
                batched = current + rng.randint(1, 512)
                successful = max(MIN_FINISHED_REQUESTS, 2 * offered)
                context_input: object = _context(
                    current_max_num_seqs=current,
                    current_max_num_batched_tokens=batched,
                    offered_concurrency=offered,
                    throttle_successful_requests=successful,
                )
                selected_base = (
                    pressure_base if mode == 1 else increase_base
                )
                window_input: object = replace(
                    _with_finished(selected_base, successful),
                    max_requests_running=current,
                    max_requests_waiting=(
                        0 if mode in {1, 2} else rng.randint(1, offered)
                    ),
                )
            elif mode == 4:
                window_input = increase_base
                context_input = f"{_PRIVATE_LABEL}-{index}"
            elif mode == 5:
                window_input = f"{_PRIVATE_LABEL}-{index}"
                context_input = _context()
            else:
                elapsed = rng.choice((4.0, 5.0, 8.0, 20.0, 20.1))
                request_count = rng.randint(1, 3_000)
                output_count = rng.randint(0, 6_000)
                observation_count = rng.randint(2, 12)
                latency_counts = tuple(
                    sorted(
                        (
                            (name, request_count)
                            if rng.randrange(3)
                            else (name, rng.randint(0, 3_000))
                        )
                        for name in ("ttft", "e2e", "queue")
                    )
                )
                scope_mode = rng.randrange(3)
                if scope_mode == 0:
                    scope_consistent, scope_count = True, 1
                elif scope_mode == 1:
                    scope_consistent, scope_count = False, rng.randint(0, 4)
                else:
                    scope_consistent, scope_count = None, None
                randomized_window = _with_elapsed(
                    _with_finished(increase_base, request_count),
                    elapsed,
                )
                window_input = replace(
                    randomized_window,
                    observations=observation_count,
                    output_tokens=output_count,
                    output_tokens_per_second=output_count / elapsed,
                    preemptions=rng.randint(0, 8),
                    mean_queue_time_ms=float(rng.randint(0, 2_500)),
                    mean_ttft_ms=float(rng.randint(0, 2_500)),
                    mean_e2e_ms=float(rng.randint(0, 2_500)),
                    max_requests_running=rng.randint(0, 1_500),
                    max_requests_waiting=rng.randint(0, 1_500),
                    max_requests_swapped=rng.choice((None, 0, 1, 2)),
                    max_kv_cache_usage_fraction=(
                        rng.randint(0, 1_000) / 1_000.0
                    ),
                    histogram_observation_counts=latency_counts,
                    metric_scope_consistent=scope_consistent,
                    metric_scope_count=scope_count,
                )
                current = rng.randint(1, 1_024)
                offered = rng.randint(1, 1_500)
                context_input = _context(
                    current_max_num_seqs=current,
                    current_max_num_batched_tokens=rng.randint(1, 1_500),
                    offered_concurrency=offered,
                    throttle_successful_requests=(
                        request_count
                        if rng.randrange(2)
                        else rng.randint(1, 3_000)
                    ),
                    throttle_failed_requests=rng.randint(0, 2),
                    effective_flags_provenance=rng.choice(
                        (
                            "unknown",
                            "operator_attested",
                            "runtime_verified",
                        )
                    ),
                    traffic_scope=rng.choice(
                        (
                            "unconfirmed",
                            "operator_attested_exclusive",
                        )
                    ),
                )

            try:
                analysis = analyze_bottleneck(
                    window_input,  # type: ignore[arg-type]
                    context_input,  # type: ignore[arg-type]
                )
            except BottleneckAnalysisError as error:
                error_count += 1
                self.assertIn(error.code, fixed_error_codes)
                self.assertEqual(str(error), error.code)
                self.assertEqual(error.args, (error.code,))
                self.assertNotIn(_PRIVATE_LABEL, str(error))
                continue

            artifact_counts[analysis.status] += 1
            public = analysis.to_public_dict()
            json.dumps(public, allow_nan=False, sort_keys=True)
            self.assertIs(public["decision_eligible"], False)
            self.assertEqual(public["decision_effect"], "none")
            self.assertIs(public["auto_apply"], False)
            suggestion = public["suggestion"]
            if suggestion is None:
                continue
            assert isinstance(suggestion, dict)
            self.assertEqual(suggestion["target_field"], "max_num_seqs")
            self.assertIs(suggestion["auto_apply"], False)
            self.assertIs(suggestion["guaranteed_outcome"], False)
            self.assertEqual(suggestion["decision_effect"], "none")
            self.assertEqual(
                {
                    key
                    for key in suggestion
                    if key.endswith("_test_value")
                },
                {"candidate_test_value"},
            )
            current_value = suggestion["current_value"]
            candidate = suggestion["candidate_test_value"]
            direction = suggestion["direction"]
            math_values = suggestion["math"]
            assert type(current_value) is int
            assert type(candidate) is int
            assert isinstance(math_values, dict)
            self.assertGreaterEqual(candidate, 1)
            self.assertNotEqual(candidate, current_value)
            step = (current_value + 3) // 4
            self.assertEqual(math_values["policy_step"], step)
            if direction == "increase":
                self.assertGreater(candidate, current_value)
                self.assertEqual(
                    candidate,
                    min(
                        current_value + step,
                        math_values["offered_concurrency_ceiling"],
                        math_values["max_num_batched_tokens_ceiling"],
                    ),
                )
            else:
                self.assertEqual(direction, "decrease")
                self.assertLess(candidate, current_value)
                self.assertEqual(candidate, max(1, current_value - step))

        self.assertEqual(sum(artifact_counts.values()) + error_count, 60_000)
        self.assertGreater(error_count, 0)
        for status, count in artifact_counts.items():
            with self.subTest(status=status, seed=hex(seed)):
                self.assertGreater(count, 0)

    def test_randomized_formula_properties_are_bounded_and_deterministic(
        self,
    ) -> None:
        rng = random.Random(0xB0771E)
        base_increase = _derived_window()
        base_decrease = _derived_window(
            preemptions=1,
            kv_fraction=HIGH_KV_FRACTION,
        )
        for iteration in range(200):
            direction = "increase" if iteration % 2 == 0 else "decrease"
            current = rng.randint(2, 10_000)
            offered = current + rng.randint(1, 1_000)
            batched = current + rng.randint(1, 1_000)
            successful = max(MIN_FINISHED_REQUESTS, 2 * offered)
            context = _context(
                current_max_num_seqs=current,
                current_max_num_batched_tokens=batched,
                offered_concurrency=offered,
                throttle_successful_requests=successful,
            )
            base = base_increase if direction == "increase" else base_decrease
            window = replace(
                _with_finished(base, successful),
                max_requests_running=current,
                max_requests_waiting=(1 if direction == "increase" else 0),
            )
            result = analyze_bottleneck(window, context)
            self.assertEqual(result.status, "suggestion_available")
            suggestion = result.suggestion
            assert suggestion is not None
            expected_step = (current + 3) // 4
            if direction == "increase":
                expected = min(
                    offered, batched, current + expected_step
                )
                self.assertGreater(expected, current)
            else:
                expected = max(1, current - expected_step)
                self.assertLess(expected, current)
            self.assertEqual(suggestion.direction, direction)
            self.assertEqual(suggestion.policy_step, expected_step)
            self.assertEqual(suggestion.candidate_max_num_seqs, expected)
            self.assertGreaterEqual(suggestion.candidate_max_num_seqs, 1)
            if direction == "increase":
                self.assertLessEqual(
                    suggestion.candidate_max_num_seqs, offered
                )
                self.assertLessEqual(
                    suggestion.candidate_max_num_seqs, batched
                )
            self.assert_hard_safety_invariants(result)

    def test_randomized_invalid_shapes_never_reflect_payloads(self) -> None:
        rng = random.Random(0xFA11C105ED)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789:/?&="
        for _ in range(100):
            private = "private-" + "".join(
                rng.choice(alphabet) for _ in range(rng.randint(8, 40))
            )
            if rng.randrange(2):
                callable_ = lambda private=private: analyze_bottleneck(
                    _derived_window(), private  # type: ignore[arg-type]
                )
                expected = "bottleneck_invalid_context"
            else:
                callable_ = lambda private=private: analyze_bottleneck(
                    private, _context()  # type: ignore[arg-type]
                )
                expected = "bottleneck_invalid_metrics_window"
            self.assert_analysis_code(
                expected, callable_, secret=private
            )


if __name__ == "__main__":
    unittest.main()
