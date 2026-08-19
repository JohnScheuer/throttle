"""Fail-closed, suggestion-only analysis of supplementary server metrics.

This module remains isolated from Throttle's benchmark runner, saved-report
schema, comparison logic, and Golden protocol.  Its output can be presented
only by the explicitly selected ``experimental-tuning`` command after an
independent safety audit.  It selects at most one bounded ``max_num_seqs``
value for a future experiment.  It never applies configuration, changes
decision eligibility, predicts savings, or claims that the proposed value will
improve an outcome.

The rules were written independently from public vLLM metric and scheduler
documentation.  In particular, this module does not estimate KV consumption
per request: the collector's running-request and KV-usage maxima are sampled
independently and are not a valid capacity estimate input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

from .server_metrics import (
    MAX_OBSERVATIONS,
    MAX_SAFE_NUMERIC_MAGNITUDE,
    MetricsCollectionError,
    MetricsWindow,
)


ANALYSIS_SCHEMA_VERSION = "1.0"
ANALYSIS_ARTIFACT_TYPE = "throttle_bottleneck_analysis"
ANALYSIS_SCOPE = "exploratory_server_metrics_analysis"
DECISION_EFFECT = "none"
CLAIM_STRENGTH = "hypothesis_only"
SUGGESTION_LABEL = "exploratory_test_suggestion_not_a_recommendation"
VALIDATION_REQUIRED = "counterbalanced_protocol_before_any_config_decision"

MIN_OBSERVATIONS = 5
MIN_ELAPSED_SECONDS = 5.0
MAX_AVERAGE_SAMPLE_SPACING_SECONDS = 5.0
MIN_FINISHED_REQUESTS = 30
LOW_KV_FRACTION = 0.70
HIGH_KV_FRACTION = 0.90
MATERIAL_QUEUE_SHARE = 0.20
POLICY_STEP_NUMERATOR = 1
POLICY_STEP_DENOMINATOR = 4
MAX_CONTEXT_INTEGER = 2_147_483_647

_PROVENANCE_VALUES = frozenset(
    {"unknown", "operator_attested", "runtime_verified"}
)
_TRAFFIC_SCOPE_VALUES = frozenset(
    {"unconfirmed", "operator_attested_exclusive"}
)
_QUALITY_REASONS = frozenset(
    {
        "effective_flags_not_runtime_verified",
        "counter_rate_inconsistent",
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
_STATUSES = frozenset(
    {"insufficient_evidence", "no_clear_signal", "suggestion_available"}
)
_DIRECTIONS = frozenset({"increase", "decrease"})
_FORMULAS = {
    "increase": "current_plus_ceil_25pct_bounded_by_verified_limits",
    "decrease": "current_minus_ceil_25pct_bounded_at_one",
}
_HYPOTHESES = {
    "increase": (
        "Testing a higher max_num_seqs may reduce observed queueing under "
        "this exact workload."
    ),
    "decrease": (
        "Testing a lower max_num_seqs may reduce observed KV pressure and "
        "preemptions under this exact workload."
    ),
}
_RISKS = {
    "increase": (
        "A higher value may increase KV pressure, preemptions, or latency."
    ),
    "decrease": (
        "A lower value may reduce throughput or increase queueing."
    ),
}
_CURRENT_GOLDEN_SUPPORT = frozenset(
    {"supported_arbitrary_pair"}
)
_NEXT_STEPS = {
    "supported_arbitrary_pair": (
        "run_generalized_golden_protocol_at_offered_concurrency"
    ),
}
_ANALYSIS_CAVEATS = (
    "This artifact contains an exploratory test suggestion, not a "
    "configuration recommendation.",
    "Exporter-to-inference-deployment matching and traffic isolation are "
    "operator-attested, not independently proven.",
    "Sampled KV and running-request maxima are not co-timed, so no "
    "per-request KV usage or fit-under capacity is inferred.",
    "The 25 percent step is a bounded search policy, not an estimated "
    "optimum or guaranteed safe capacity.",
    "Server-side mean latency cannot establish client-side p95 SLO "
    "compliance or latency parity.",
    "Exporter completions are usable only when every finish reason is "
    "classified as stop or length; all other outcomes fail closed.",
    "Throttle's generalized Golden command can validate this pair at one "
    "explicit closed-loop concurrency; reaching that client load is not proof "
    "of direct server-scheduler saturation.",
    "No cost or savings conclusion can be derived from this analysis.",
)

_REQUIRED_SOURCE_GROUPS = (
    ("vllm:request_success_total",),
    ("vllm:generation_tokens_total",),
    ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    ("vllm:num_requests_running",),
    ("vllm:num_requests_waiting",),
    ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    ("vllm:time_to_first_token_seconds_sum",),
    ("vllm:time_to_first_token_seconds_count",),
    ("vllm:e2e_request_latency_seconds_sum",),
    ("vllm:e2e_request_latency_seconds_count",),
    ("vllm:request_queue_time_seconds_sum",),
    ("vllm:request_queue_time_seconds_count",),
)
_REQUIRED_SINGLE_SERIES_KEYS = frozenset(
    {
        "output_tokens",
        "preemptions",
        "requests_running",
        "requests_waiting",
        "kv_cache_usage",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
    }
)


class BottleneckAnalysisError(ValueError):
    """A fixed-code analysis failure that never reflects caller data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> NoReturn:
    raise BottleneckAnalysisError(code)


@dataclass(frozen=True, kw_only=True, slots=True)
class BottleneckContext:
    """Explicit, operator-supplied context for one metrics window."""

    current_max_num_seqs: int
    current_max_num_batched_tokens: int
    offered_concurrency: int
    throttle_successful_requests: int
    throttle_failed_requests: int
    effective_flags_provenance: str
    traffic_scope: str

    def __post_init__(self) -> None:
        _validate_context(self)


@dataclass(frozen=True, slots=True)
class BottleneckSignals:
    """Allowlisted diagnostics used by the fixed analysis rules."""

    elapsed_seconds: float
    observations: int
    average_sample_spacing_seconds: float
    sampled_kv_headroom_fraction: float
    queue_share_of_mean_e2e: float
    preemptions_per_100_finished: float
    max_requests_running: int
    max_requests_waiting: int
    max_kv_cache_usage_fraction: float
    preemptions: int
    mean_queue_time_ms: float
    mean_ttft_ms: float
    mean_e2e_ms: float
    output_tokens: int
    output_tokens_per_second: float
    finished_requests: int
    finished_requests_per_second: float
    allowed_finished_requests: int
    disallowed_finished_requests: int
    unclassified_finished_requests: int
    latency_observations: int
    max_requests_swapped: int | None
    current_max_num_seqs: int
    current_max_num_batched_tokens: int
    offered_concurrency: int
    configured_limit_exercised: bool

    def __post_init__(self) -> None:
        _validate_signals(self)

    def to_public_dict(self) -> dict[str, object]:
        _validate_signals(self)
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "observations": self.observations,
            "average_sample_spacing_seconds": (
                self.average_sample_spacing_seconds
            ),
            "sampled_kv_headroom_fraction": (
                self.sampled_kv_headroom_fraction
            ),
            "queue_share_of_mean_e2e": self.queue_share_of_mean_e2e,
            "preemptions_per_100_finished": (
                self.preemptions_per_100_finished
            ),
            "max_requests_running": self.max_requests_running,
            "max_requests_waiting": self.max_requests_waiting,
            "max_kv_cache_usage_fraction": (
                self.max_kv_cache_usage_fraction
            ),
            "preemptions": self.preemptions,
            "mean_queue_time_ms": self.mean_queue_time_ms,
            "mean_ttft_ms": self.mean_ttft_ms,
            "mean_e2e_ms": self.mean_e2e_ms,
            "output_tokens": self.output_tokens,
            "output_tokens_per_second": self.output_tokens_per_second,
            "finished_requests": self.finished_requests,
            "finished_requests_per_second": (
                self.finished_requests_per_second
            ),
            "allowed_finished_requests": self.allowed_finished_requests,
            "disallowed_finished_requests": (
                self.disallowed_finished_requests
            ),
            "unclassified_finished_requests": (
                self.unclassified_finished_requests
            ),
            "latency_observations": self.latency_observations,
            "max_requests_swapped": self.max_requests_swapped,
            "current_max_num_seqs": self.current_max_num_seqs,
            "current_max_num_batched_tokens": (
                self.current_max_num_batched_tokens
            ),
            "offered_concurrency": self.offered_concurrency,
            "configured_limit_exercised": (
                self.configured_limit_exercised
            ),
        }


@dataclass(frozen=True, slots=True)
class BottleneckSuggestion:
    """One bounded, single-flag candidate for a future experiment."""

    direction: str
    current_max_num_seqs: int
    candidate_max_num_seqs: int
    policy_step: int
    offered_concurrency: int
    current_max_num_batched_tokens: int
    formula_id: str
    hypothesis: str
    risk: str
    current_golden_support: str
    required_next_step: str
    label: str = SUGGESTION_LABEL
    target_field: str = "max_num_seqs"
    claim_strength: str = CLAIM_STRENGTH
    validation_required: str = VALIDATION_REQUIRED
    decision_effect: str = DECISION_EFFECT
    auto_apply: bool = False
    guaranteed_outcome: bool = False

    def __post_init__(self) -> None:
        _validate_suggestion(self)

    def to_public_dict(self) -> dict[str, object]:
        _validate_suggestion(self)
        return {
            "label": self.label,
            "target_field": self.target_field,
            "direction": self.direction,
            "current_value": self.current_max_num_seqs,
            "candidate_test_value": self.candidate_max_num_seqs,
            "formula_id": self.formula_id,
            "math": {
                "policy_step_numerator": POLICY_STEP_NUMERATOR,
                "policy_step_denominator": POLICY_STEP_DENOMINATOR,
                "policy_step": self.policy_step,
                "offered_concurrency_ceiling": self.offered_concurrency,
                "max_num_batched_tokens_ceiling": (
                    self.current_max_num_batched_tokens
                ),
            },
            "hypothesis": self.hypothesis,
            "risk": self.risk,
            "claim_strength": self.claim_strength,
            "validation_required": self.validation_required,
            "current_golden_support": self.current_golden_support,
            "required_next_step": self.required_next_step,
            "decision_effect": self.decision_effect,
            "auto_apply": self.auto_apply,
            "guaranteed_outcome": self.guaranteed_outcome,
        }


@dataclass(frozen=True, slots=True)
class BottleneckAnalysis:
    """Validated public projection of an isolated exploratory analysis."""

    status: str
    quality_reasons: tuple[str, ...]
    no_suggestion_reasons: tuple[str, ...]
    signals: BottleneckSignals | None
    suggestion: BottleneckSuggestion | None
    schema_version: str = ANALYSIS_SCHEMA_VERSION
    artifact_type: str = ANALYSIS_ARTIFACT_TYPE
    scope: str = ANALYSIS_SCOPE
    decision_eligible: bool = False
    decision_effect: str = DECISION_EFFECT
    auto_apply: bool = False
    caveats: tuple[str, ...] = _ANALYSIS_CAVEATS

    def __post_init__(self) -> None:
        _validate_analysis(self)

    def to_public_dict(self) -> dict[str, object]:
        _validate_analysis(self)
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "scope": self.scope,
            "status": self.status,
            "decision_eligible": self.decision_eligible,
            "decision_effect": self.decision_effect,
            "auto_apply": self.auto_apply,
            "quality_reasons": list(self.quality_reasons),
            "no_suggestion_reasons": list(
                self.no_suggestion_reasons
            ),
            "signals": (
                None if self.signals is None
                else BottleneckSignals.to_public_dict(self.signals)
            ),
            "suggestion": (
                None if self.suggestion is None
                else BottleneckSuggestion.to_public_dict(self.suggestion)
            ),
            "caveats": list(self.caveats),
        }


def _is_bounded_int(value: object, *, allow_zero: bool = False) -> bool:
    minimum = 0 if allow_zero else 1
    return (
        type(value) is int
        and minimum <= value <= MAX_CONTEXT_INTEGER
    )


def _is_bounded_float(
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float = float(MAX_SAFE_NUMERIC_MAGNITUDE),
) -> bool:
    return (
        type(value) is float
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def _validate_context(context: BottleneckContext) -> None:
    if (
        type(context) is not BottleneckContext
        or not _is_bounded_int(context.current_max_num_seqs)
        or not _is_bounded_int(context.current_max_num_batched_tokens)
        or not _is_bounded_int(context.offered_concurrency)
        or not _is_bounded_int(context.throttle_successful_requests)
        or not _is_bounded_int(
            context.throttle_failed_requests,
            allow_zero=True,
        )
        or type(context.effective_flags_provenance) is not str
        or context.effective_flags_provenance not in _PROVENANCE_VALUES
        or type(context.traffic_scope) is not str
        or context.traffic_scope not in _TRAFFIC_SCOPE_VALUES
    ):
        _fail("bottleneck_invalid_context")


def _validate_signals(signals: BottleneckSignals) -> None:
    if type(signals) is not BottleneckSignals:
        _fail("bottleneck_invalid_output")
    fractions = (
        signals.sampled_kv_headroom_fraction,
        signals.queue_share_of_mean_e2e,
        signals.max_kv_cache_usage_fraction,
    )
    nonnegative_floats = (
        signals.elapsed_seconds,
        signals.average_sample_spacing_seconds,
        signals.preemptions_per_100_finished,
        signals.mean_queue_time_ms,
        signals.mean_ttft_ms,
        signals.mean_e2e_ms,
        signals.output_tokens_per_second,
        signals.finished_requests_per_second,
    )
    positive_integers = (
        signals.observations,
        signals.finished_requests,
        signals.latency_observations,
        signals.output_tokens,
        signals.current_max_num_seqs,
        signals.current_max_num_batched_tokens,
        signals.offered_concurrency,
    )
    nonnegative_integers = (
        signals.max_requests_running,
        signals.max_requests_waiting,
        signals.preemptions,
        signals.allowed_finished_requests,
        signals.disallowed_finished_requests,
        signals.unclassified_finished_requests,
    )
    if (
        any(
            not _is_bounded_float(value, minimum=0.0, maximum=1.0)
            for value in fractions
        )
        or any(
            not _is_bounded_float(value)
            for value in nonnegative_floats
        )
        or any(not _is_bounded_int(value) for value in positive_integers)
        or any(
            not _is_bounded_int(value, allow_zero=True)
            for value in nonnegative_integers
        )
        or (
            signals.max_requests_swapped is not None
            and not _is_bounded_int(
                signals.max_requests_swapped,
                allow_zero=True,
            )
        )
        or type(signals.configured_limit_exercised) is not bool
    ):
        _fail("bottleneck_invalid_output")
    if (
        signals.observations < 2
        or signals.observations > MAX_OBSERVATIONS + 2
        or signals.elapsed_seconds <= 0.0
        or signals.average_sample_spacing_seconds <= 0.0
        or signals.mean_e2e_ms <= 0.0
        or signals.output_tokens_per_second <= 0.0
        or signals.finished_requests_per_second <= 0.0
        or signals.observations < MIN_OBSERVATIONS
        or signals.elapsed_seconds < MIN_ELAPSED_SECONDS
        or signals.average_sample_spacing_seconds
        > MAX_AVERAGE_SAMPLE_SPACING_SECONDS
        or signals.finished_requests < MIN_FINISHED_REQUESTS
        or signals.finished_requests < 2 * signals.offered_concurrency
        or signals.latency_observations != signals.finished_requests
        or signals.allowed_finished_requests != signals.finished_requests
        or signals.disallowed_finished_requests != 0
        or signals.unclassified_finished_requests != 0
        or signals.output_tokens < signals.finished_requests
        or not (
            0.0
            <= signals.mean_queue_time_ms
            <= signals.mean_ttft_ms
            <= signals.mean_e2e_ms
        )
        or signals.current_max_num_batched_tokens
        < signals.current_max_num_seqs
        or signals.max_requests_running > signals.current_max_num_seqs
        or signals.max_requests_running > signals.offered_concurrency
        or signals.max_requests_waiting > signals.offered_concurrency
    ):
        _fail("bottleneck_invalid_output")
    expected_spacing = signals.elapsed_seconds / (signals.observations - 1)
    expected_headroom = 1.0 - signals.max_kv_cache_usage_fraction
    expected_queue_share = signals.mean_queue_time_ms / signals.mean_e2e_ms
    expected_preemption_density = (
        100.0 * signals.preemptions / signals.finished_requests
    )
    expected_finished_rate = (
        signals.finished_requests / signals.elapsed_seconds
    )
    expected_output_rate = signals.output_tokens / signals.elapsed_seconds
    expected_limit_exercised = (
        signals.offered_concurrency >= signals.current_max_num_seqs
        and signals.max_requests_running == signals.current_max_num_seqs
    )
    if (
        signals.average_sample_spacing_seconds != expected_spacing
        or signals.sampled_kv_headroom_fraction != expected_headroom
        or signals.queue_share_of_mean_e2e != expected_queue_share
        or signals.preemptions_per_100_finished
        != expected_preemption_density
        or signals.finished_requests_per_second != expected_finished_rate
        or signals.output_tokens_per_second != expected_output_rate
        or signals.configured_limit_exercised
        is not expected_limit_exercised
    ):
        _fail("bottleneck_invalid_output")


def _golden_support(current: int, candidate: int) -> str:
    del current, candidate
    return "supported_arbitrary_pair"


def _validate_suggestion(suggestion: BottleneckSuggestion) -> None:
    if (
        type(suggestion) is not BottleneckSuggestion
        or type(suggestion.direction) is not str
        or suggestion.direction not in _DIRECTIONS
        or not _is_bounded_int(suggestion.current_max_num_seqs)
        or not _is_bounded_int(suggestion.candidate_max_num_seqs)
        or not _is_bounded_int(suggestion.policy_step)
        or not _is_bounded_int(suggestion.offered_concurrency)
        or not _is_bounded_int(
            suggestion.current_max_num_batched_tokens
        )
        or type(suggestion.formula_id) is not str
        or suggestion.formula_id != _FORMULAS[suggestion.direction]
        or type(suggestion.hypothesis) is not str
        or suggestion.hypothesis != _HYPOTHESES[suggestion.direction]
        or type(suggestion.risk) is not str
        or suggestion.risk != _RISKS[suggestion.direction]
        or type(suggestion.current_golden_support) is not str
        or suggestion.current_golden_support
        not in _CURRENT_GOLDEN_SUPPORT
        or type(suggestion.required_next_step) is not str
        or suggestion.required_next_step
        != _NEXT_STEPS[suggestion.current_golden_support]
        or type(suggestion.label) is not str
        or suggestion.label != SUGGESTION_LABEL
        or type(suggestion.target_field) is not str
        or suggestion.target_field != "max_num_seqs"
        or type(suggestion.claim_strength) is not str
        or suggestion.claim_strength != CLAIM_STRENGTH
        or type(suggestion.validation_required) is not str
        or suggestion.validation_required != VALIDATION_REQUIRED
        or type(suggestion.decision_effect) is not str
        or suggestion.decision_effect != DECISION_EFFECT
        or type(suggestion.auto_apply) is not bool
        or suggestion.auto_apply
        or type(suggestion.guaranteed_outcome) is not bool
        or suggestion.guaranteed_outcome
    ):
        _fail("bottleneck_invalid_output")
    current = suggestion.current_max_num_seqs
    expected_step = (current + POLICY_STEP_DENOMINATOR - 1) // (
        POLICY_STEP_DENOMINATOR
    )
    if suggestion.policy_step != expected_step:
        _fail("bottleneck_invalid_output")
    if suggestion.direction == "increase":
        expected_candidate = min(
            suggestion.offered_concurrency,
            suggestion.current_max_num_batched_tokens,
            current + expected_step,
        )
        direction_valid = suggestion.candidate_max_num_seqs > current
    else:
        expected_candidate = max(1, current - expected_step)
        direction_valid = suggestion.candidate_max_num_seqs < current
    if (
        suggestion.candidate_max_num_seqs != expected_candidate
        or not direction_valid
        or suggestion.current_golden_support
        != _golden_support(current, suggestion.candidate_max_num_seqs)
    ):
        _fail("bottleneck_invalid_output")


def _validate_reason_tuple(
    values: tuple[str, ...], allowed: frozenset[str]
) -> bool:
    return (
        type(values) is tuple
        and len(values) <= len(allowed)
        and all(type(value) is str and value in allowed for value in values)
        and values == tuple(sorted(set(values)))
    )


def _validate_analysis(analysis: BottleneckAnalysis) -> None:
    if (
        type(analysis) is not BottleneckAnalysis
        or type(analysis.status) is not str
        or analysis.status not in _STATUSES
        or not _validate_reason_tuple(
            analysis.quality_reasons,
            _QUALITY_REASONS,
        )
        or not _validate_reason_tuple(
            analysis.no_suggestion_reasons,
            _NO_SUGGESTION_REASONS,
        )
        or type(analysis.schema_version) is not str
        or analysis.schema_version != ANALYSIS_SCHEMA_VERSION
        or type(analysis.artifact_type) is not str
        or analysis.artifact_type != ANALYSIS_ARTIFACT_TYPE
        or type(analysis.scope) is not str
        or analysis.scope != ANALYSIS_SCOPE
        or type(analysis.decision_eligible) is not bool
        or analysis.decision_eligible
        or type(analysis.decision_effect) is not str
        or analysis.decision_effect != DECISION_EFFECT
        or type(analysis.auto_apply) is not bool
        or analysis.auto_apply
        or type(analysis.caveats) is not tuple
        or len(analysis.caveats) != len(_ANALYSIS_CAVEATS)
        or any(type(value) is not str for value in analysis.caveats)
        or analysis.caveats != _ANALYSIS_CAVEATS
    ):
        _fail("bottleneck_invalid_output")
    if analysis.status == "insufficient_evidence":
        valid_shape = (
            bool(analysis.quality_reasons)
            and not analysis.no_suggestion_reasons
            and analysis.signals is None
            and analysis.suggestion is None
        )
    elif analysis.status == "no_clear_signal":
        valid_shape = (
            not analysis.quality_reasons
            and bool(analysis.no_suggestion_reasons)
            and type(analysis.signals) is BottleneckSignals
            and analysis.suggestion is None
        )
    else:
        valid_shape = (
            not analysis.quality_reasons
            and not analysis.no_suggestion_reasons
            and type(analysis.signals) is BottleneckSignals
            and type(analysis.suggestion) is BottleneckSuggestion
        )
    if not valid_shape:
        _fail("bottleneck_invalid_output")
    if analysis.signals is not None:
        _validate_signals(analysis.signals)
    if analysis.suggestion is not None:
        _validate_suggestion(analysis.suggestion)
    if analysis.signals is not None:
        expected_reasons, expected_suggestion = _evaluate_signals(
            analysis.signals
        )
        if analysis.status == "no_clear_signal":
            if (
                analysis.no_suggestion_reasons != expected_reasons
                or expected_suggestion is not None
            ):
                _fail("bottleneck_invalid_output")
        elif analysis.status == "suggestion_available":
            if (
                expected_reasons
                or analysis.suggestion != expected_suggestion
            ):
                _fail("bottleneck_invalid_output")


def _has_required_source(window: MetricsWindow) -> bool:
    sources = set(window.source_families)
    return all(
        sum(name in sources for name in group) == 1
        for group in _REQUIRED_SOURCE_GROUPS
    )


def _quality_reasons(
    window: MetricsWindow,
    context: BottleneckContext,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if context.effective_flags_provenance != "runtime_verified":
        reasons.add("effective_flags_not_runtime_verified")
    if context.traffic_scope != "operator_attested_exclusive":
        reasons.add("traffic_scope_not_exclusive")
    if context.throttle_failed_requests != 0:
        reasons.add("throttle_requests_failed")
    if (
        context.current_max_num_batched_tokens
        < context.current_max_num_seqs
    ):
        reasons.add("max_num_batched_tokens_below_max_num_seqs")
    if window.elapsed_seconds < MIN_ELAPSED_SECONDS:
        reasons.add("window_too_short")
    if window.observations < MIN_OBSERVATIONS:
        reasons.add("too_few_observations")
    if window.observations >= 2:
        spacing = window.elapsed_seconds / (window.observations - 1)
        if spacing > MAX_AVERAGE_SAMPLE_SPACING_SECONDS:
            reasons.add("sample_spacing_too_sparse")
    if (
        window.requests_finished is None
        or window.requests_finished < MIN_FINISHED_REQUESTS
        or window.requests_finished
        < 2 * context.offered_concurrency
    ):
        reasons.add("too_few_finished_requests")
    if window.requests_finished != context.throttle_successful_requests:
        reasons.add("exporter_request_count_mismatch")
    if (
        window.allowed_finished_requests is None
        or window.disallowed_finished_requests is None
        or window.unclassified_finished_requests is None
        or window.allowed_finished_requests != window.requests_finished
        or window.disallowed_finished_requests != 0
        or window.unclassified_finished_requests != 0
    ):
        reasons.add("finish_reason_evidence_not_clean")
    required_values = (
        window.requests_finished,
        window.output_tokens,
        window.preemptions,
        window.finished_requests_per_second,
        window.output_tokens_per_second,
        window.mean_queue_time_ms,
        window.mean_ttft_ms,
        window.mean_e2e_ms,
        window.max_requests_running,
        window.max_requests_waiting,
        window.max_kv_cache_usage_fraction,
    )
    if any(value is None for value in required_values):
        reasons.add("missing_required_metric")
    if (
        window.output_tokens is not None
        and window.output_tokens <= 0
    ) or (
        window.output_tokens_per_second is not None
        and window.output_tokens_per_second <= 0.0
    ):
        reasons.add("nonpositive_output_evidence")
    if (
        window.output_tokens is not None
        and window.requests_finished is not None
        and window.output_tokens < window.requests_finished
    ):
        reasons.add("output_token_count_below_finished_requests")
    rate_checks = (
        (
            window.requests_finished,
            window.finished_requests_per_second,
        ),
        (window.output_tokens, window.output_tokens_per_second),
        (window.prompt_tokens, window.prompt_tokens_per_second),
    )
    for count, rate in rate_checks:
        if (count is None) != (rate is None):
            reasons.add("counter_rate_inconsistent")
        elif count is not None and rate is not None:
            expected_rate = count / window.elapsed_seconds
            if not math.isclose(
                rate,
                expected_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                reasons.add("counter_rate_inconsistent")
    if not _has_required_source(window):
        reasons.add("required_metric_source_missing")
    if (
        window.metric_scope_consistent is not True
        or window.metric_scope_count != 1
    ):
        reasons.add("metric_scope_not_single_consistent")
    counts = dict(window.series_counts)
    if any(counts.get(name) != 1 for name in _REQUIRED_SINGLE_SERIES_KEYS):
        reasons.add("required_metric_series_count_not_one")
    running = window.max_requests_running
    waiting = window.max_requests_waiting
    if (
        running is not None
        and running > context.current_max_num_seqs
    ):
        reasons.add("running_exceeds_current_max_num_seqs")
    if (
        (running is not None and running > context.offered_concurrency)
        or (
            waiting is not None
            and waiting > context.offered_concurrency
        )
    ):
        reasons.add("gauge_exceeds_offered_concurrency")
    queue = window.mean_queue_time_ms
    ttft = window.mean_ttft_ms
    e2e = window.mean_e2e_ms
    observation_counts = dict(window.histogram_observation_counts)
    if (
        window.requests_finished is None
        or any(
            observation_counts.get(name) != window.requests_finished
            for name in ("ttft", "e2e", "queue")
        )
    ):
        reasons.add("latency_observation_count_mismatch")
    if (
        queue is not None
        and ttft is not None
        and e2e is not None
        and not (0.0 <= queue <= ttft <= e2e and e2e > 0.0)
    ):
        reasons.add("latency_metrics_inconsistent")
    return tuple(sorted(reasons))


def _derive_signals(
    window: MetricsWindow,
    context: BottleneckContext,
) -> BottleneckSignals:
    finished = window.requests_finished
    preemptions = window.preemptions
    running = window.max_requests_running
    waiting = window.max_requests_waiting
    kv = window.max_kv_cache_usage_fraction
    queue = window.mean_queue_time_ms
    ttft = window.mean_ttft_ms
    e2e = window.mean_e2e_ms
    output_rate = window.output_tokens_per_second
    output_tokens = window.output_tokens
    finished_rate = window.finished_requests_per_second
    allowed_finished = window.allowed_finished_requests
    disallowed_finished = window.disallowed_finished_requests
    unclassified_finished = window.unclassified_finished_requests
    latency_observations = dict(
        window.histogram_observation_counts
    ).get("e2e")
    if any(
        value is None
        for value in (
            finished,
            preemptions,
            running,
            waiting,
            kv,
            queue,
            ttft,
            e2e,
            output_rate,
            output_tokens,
            finished_rate,
            allowed_finished,
            disallowed_finished,
            unclassified_finished,
            latency_observations,
        )
    ):
        _fail("bottleneck_invalid_metrics_window")
    assert finished is not None
    assert preemptions is not None
    assert running is not None
    assert waiting is not None
    assert kv is not None
    assert queue is not None
    assert ttft is not None
    assert e2e is not None
    assert output_rate is not None
    assert output_tokens is not None
    assert finished_rate is not None
    assert allowed_finished is not None
    assert disallowed_finished is not None
    assert unclassified_finished is not None
    assert latency_observations is not None
    return BottleneckSignals(
        elapsed_seconds=window.elapsed_seconds,
        observations=window.observations,
        average_sample_spacing_seconds=(
            window.elapsed_seconds / (window.observations - 1)
        ),
        sampled_kv_headroom_fraction=1.0 - kv,
        queue_share_of_mean_e2e=queue / e2e,
        preemptions_per_100_finished=(
            100.0 * preemptions / finished
        ),
        max_requests_running=running,
        max_requests_waiting=waiting,
        max_kv_cache_usage_fraction=kv,
        preemptions=preemptions,
        mean_queue_time_ms=queue,
        mean_ttft_ms=ttft,
        mean_e2e_ms=e2e,
        output_tokens=output_tokens,
        output_tokens_per_second=output_rate,
        finished_requests=finished,
        finished_requests_per_second=finished_rate,
        allowed_finished_requests=allowed_finished,
        disallowed_finished_requests=disallowed_finished,
        unclassified_finished_requests=unclassified_finished,
        latency_observations=latency_observations,
        max_requests_swapped=window.max_requests_swapped,
        current_max_num_seqs=context.current_max_num_seqs,
        current_max_num_batched_tokens=(
            context.current_max_num_batched_tokens
        ),
        offered_concurrency=context.offered_concurrency,
        configured_limit_exercised=(
            context.offered_concurrency >= context.current_max_num_seqs
            and running == context.current_max_num_seqs
        ),
    )


def _no_suggestion_reasons(
    signals: BottleneckSignals,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if not signals.configured_limit_exercised:
        reasons.add("configured_limit_not_exercised")
    if signals.max_requests_swapped is not None:
        if signals.max_requests_swapped > 0:
            reasons.add("legacy_swap_pressure_observed")
    pressure_complete = (
        signals.preemptions > 0
        and signals.max_kv_cache_usage_fraction >= HIGH_KV_FRACTION
    )
    queue_complete = (
        signals.preemptions == 0
        and signals.max_requests_waiting > 0
        and signals.queue_share_of_mean_e2e >= MATERIAL_QUEUE_SHARE
        and signals.max_kv_cache_usage_fraction <= LOW_KV_FRACTION
        and signals.offered_concurrency > signals.current_max_num_seqs
    )
    if pressure_complete or queue_complete:
        return tuple(sorted(reasons))
    if (
        (signals.preemptions > 0)
        != (
            signals.max_kv_cache_usage_fraction
            >= HIGH_KV_FRACTION
        )
    ):
        reasons.add("pressure_signal_incomplete")
    if signals.max_requests_waiting == 0:
        reasons.add("no_material_queue")
    elif signals.queue_share_of_mean_e2e < MATERIAL_QUEUE_SHARE:
        reasons.add("queue_share_below_policy_threshold")
    if (
        signals.max_requests_waiting > 0
        and signals.max_kv_cache_usage_fraction > LOW_KV_FRACTION
    ):
        reasons.add("queue_and_kv_signals_conflict")
    if (
        LOW_KV_FRACTION
        < signals.max_kv_cache_usage_fraction
        < HIGH_KV_FRACTION
    ):
        reasons.add("kv_headroom_not_clear")
    if not reasons:
        reasons.add("kv_headroom_not_clear")
    return tuple(sorted(reasons))


def _suggestion(
    direction: str,
    signals: BottleneckSignals,
) -> BottleneckSuggestion | None:
    current = signals.current_max_num_seqs
    step = (current + POLICY_STEP_DENOMINATOR - 1) // (
        POLICY_STEP_DENOMINATOR
    )
    if direction == "increase":
        candidate = min(
            signals.offered_concurrency,
            signals.current_max_num_batched_tokens,
            current + step,
        )
        if candidate <= current:
            return None
    else:
        candidate = max(1, current - step)
        if candidate >= current:
            return None
    support = _golden_support(current, candidate)
    return BottleneckSuggestion(
        direction=direction,
        current_max_num_seqs=current,
        candidate_max_num_seqs=candidate,
        policy_step=step,
        offered_concurrency=signals.offered_concurrency,
        current_max_num_batched_tokens=(
            signals.current_max_num_batched_tokens
        ),
        formula_id=_FORMULAS[direction],
        hypothesis=_HYPOTHESES[direction],
        risk=_RISKS[direction],
        current_golden_support=support,
        required_next_step=_NEXT_STEPS[support],
    )


def _evaluate_signals(
    signals: BottleneckSignals,
) -> tuple[tuple[str, ...], BottleneckSuggestion | None]:
    reasons = _no_suggestion_reasons(signals)
    if reasons:
        return reasons, None
    direction = "decrease" if signals.preemptions > 0 else "increase"
    suggestion = _suggestion(direction, signals)
    if suggestion is not None:
        return (), suggestion
    reason = (
        "no_lower_candidate"
        if direction == "decrease"
        else "no_higher_candidate_within_verified_bounds"
    )
    return (reason,), None


def analyze_bottleneck(
    window: MetricsWindow,
    context: BottleneckContext,
) -> BottleneckAnalysis:
    """Return one bounded test suggestion or fixed unavailable reasons."""

    if type(context) is not BottleneckContext:
        _fail("bottleneck_invalid_context")
    _validate_context(context)
    if type(window) is not MetricsWindow:
        _fail("bottleneck_invalid_metrics_window")
    try:
        MetricsWindow.to_public_dict(window)
    except MetricsCollectionError:
        _fail("bottleneck_invalid_metrics_window")
    quality_reasons = _quality_reasons(window, context)
    if quality_reasons:
        return BottleneckAnalysis(
            status="insufficient_evidence",
            quality_reasons=quality_reasons,
            no_suggestion_reasons=(),
            signals=None,
            suggestion=None,
        )
    signals = _derive_signals(window, context)
    no_suggestion, suggestion = _evaluate_signals(signals)
    if suggestion is None:
        return BottleneckAnalysis(
            status="no_clear_signal",
            quality_reasons=(),
            no_suggestion_reasons=no_suggestion,
            signals=signals,
            suggestion=None,
        )
    return BottleneckAnalysis(
        status="suggestion_available",
        quality_reasons=(),
        no_suggestion_reasons=(),
        signals=signals,
        suggestion=suggestion,
    )


__all__ = [
    "ANALYSIS_ARTIFACT_TYPE",
    "ANALYSIS_SCHEMA_VERSION",
    "BottleneckAnalysis",
    "BottleneckAnalysisError",
    "BottleneckContext",
    "BottleneckSignals",
    "BottleneckSuggestion",
    "HIGH_KV_FRACTION",
    "LOW_KV_FRACTION",
    "MATERIAL_QUEUE_SHARE",
    "MAX_AVERAGE_SAMPLE_SPACING_SECONDS",
    "MIN_ELAPSED_SECONDS",
    "MIN_FINISHED_REQUESTS",
    "MIN_OBSERVATIONS",
    "analyze_bottleneck",
]
