"""Independent fail-closed boundary for the experimental agent chain.

This module audits the supplementary server-metrics window and bottleneck
analysis before the explicitly selected ``experimental-tuning`` command may
present a detached projection.  It remains isolated from standard run-report
schemas, comparison code, and the Golden implementation.  Passing this
boundary never changes decision eligibility, applies a configuration, or
proves that a Golden run has taken place.

The checks below intentionally replay the analyzer policy rather than merely
trusting its validator.  The returned object owns a detached canonical copy;
it never retains a caller-owned metrics, context, or analysis object.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import NoReturn

from .bottleneck_analysis import (
    BottleneckAnalysis,
    BottleneckContext,
    BottleneckSuggestion,
)
from .server_metrics import MetricsWindow


SAFETY_VALIDATION_SCHEMA_VERSION = "1.0"
SAFETY_VALIDATION_ARTIFACT_TYPE = "throttle_agent_safety_validation"
SAFETY_VALIDATION_SCOPE = "isolated_preintegration_validation"
MAX_GOLDEN_MAX_NUM_SEQS = 2_147_483_647
MAX_CANONICAL_PROJECTION_BYTES = 65_536

_ANALYSIS_SCHEMA_VERSION = "1.0"
_ANALYSIS_ARTIFACT_TYPE = "throttle_bottleneck_analysis"
_ANALYSIS_SCOPE = "exploratory_server_metrics_analysis"
_DECISION_EFFECT = "none"
_CLAIM_STRENGTH = "hypothesis_only"
_SUGGESTION_LABEL = "exploratory_test_suggestion_not_a_recommendation"
_VALIDATION_REQUIRED = "counterbalanced_protocol_before_any_config_decision"
_GOLDEN_SUPPORT = "supported_arbitrary_pair"
_GOLDEN_NEXT_STEP = "run_generalized_golden_protocol_at_offered_concurrency"
# These reviewed policy literals are intentionally duplicated. Importing the
# analyzer or collector constants would let an accidental upstream weakening
# silently weaken both layers at once. Tests make any policy drift explicit.
_MIN_OBSERVATIONS = 5
_MIN_ELAPSED_SECONDS = 5.0
_MAX_AVERAGE_SAMPLE_SPACING_SECONDS = 5.0
_MIN_FINISHED_REQUESTS = 30
_LOW_KV_FRACTION = 0.70
_HIGH_KV_FRACTION = 0.90
_MATERIAL_QUEUE_SHARE = 0.20
_MAX_OBSERVATIONS = 1_024
_MAX_SAFE_NUMERIC_MAGNITUDE = 9_007_199_254_740_991

_PROVENANCE_VALUES = frozenset(
    {"unknown", "operator_attested", "runtime_verified"}
)
_TRAFFIC_SCOPE_VALUES = frozenset(
    {"unconfirmed", "operator_attested_exclusive"}
)
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
_STATUSES = frozenset(
    {"insufficient_evidence", "no_clear_signal", "suggestion_available"}
)
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
    "explicit closed-loop concurrency; reaching that client load is not "
    "proof of direct server-scheduler saturation.",
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
_AUDIT_CHECKS = (
    "collector_projection_validated_unbound",
    "analysis_projection_validated_unbound",
    "analysis_replayed_independently",
    "analysis_bound_to_window_and_context",
    "decision_and_action_flags_locked_false",
    "suggestion_language_allowlisted",
    "golden_handoff_requires_separate_counterbalanced_run",
    "canonical_projection_detached_from_inputs",
)
_SAFETY_CAVEATS = (
    "This result cannot authorize its own routing or insertion into a "
    "standard Throttle report; only the explicit experimental-tuning "
    "command may present the detached projection.",
    "Neither the upstream analysis nor this validation is decision-grade.",
    "A suggested pair still requires a separate six-position "
    "counterbalanced Golden run before any configuration decision.",
    "Reaching the declared client concurrency proves offered demand, not "
    "direct server-scheduler saturation.",
    "No configuration was applied and no outcome, safety, optimum, or "
    "savings is guaranteed.",
)
_AUDIT_MARKER = object()
_SEAL_DOMAIN = b"throttle-agent-safety-validation-v1\x00"


class SafetyValidationError(ValueError):
    """A fixed-code failure that never reflects caller-controlled data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> NoReturn:
    raise SafetyValidationError(code)


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        _fail("safety_projection_invariant_failed")
    if len(encoded) > MAX_CANONICAL_PROJECTION_BYTES:
        _fail("safety_projection_invariant_failed")
    return encoded


def _strict_json_object(encoded: bytes) -> dict[str, object]:
    if type(encoded) is not bytes or len(encoded) > MAX_CANONICAL_PROJECTION_BYTES:
        _fail("safety_projection_invariant_failed")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if type(key) is not str or key in result:
                _fail("safety_projection_invariant_failed")
            result[key] = value
        return result

    try:
        value = json.loads(
            encoded,
            object_pairs_hook=pairs,
            parse_constant=lambda _: _fail(
                "safety_projection_invariant_failed"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        _fail("safety_projection_invariant_failed")
    if type(value) is not dict or _canonical_json(value) != encoded:
        _fail("safety_projection_invariant_failed")
    return value


def _seal_payload(
    analysis_json: bytes,
    analysis_status: str,
    current: int | None,
    candidate: int | None,
    concurrency: int | None,
) -> bytes:
    metadata = _canonical_json(
        {
            "analysis_status": analysis_status,
            "candidate": candidate,
            "concurrency": concurrency,
            "current": current,
        }
    )
    return hashlib.sha256(_SEAL_DOMAIN + metadata + analysis_json).digest()


@dataclass(frozen=True, slots=True, init=False)
class ValidatedAgentOutputs:
    """Detached result of the safety audit; never a decision-grade artifact."""

    analysis_status: str
    current_max_num_seqs: int | None
    candidate_max_num_seqs: int | None
    offered_concurrency: int | None
    schema_version: str = SAFETY_VALIDATION_SCHEMA_VERSION
    artifact_type: str = SAFETY_VALIDATION_ARTIFACT_TYPE
    scope: str = SAFETY_VALIDATION_SCOPE
    safety_validated: bool = True
    supplementary_content_validated: bool = True
    decision_eligible: bool = False
    decision_effect: str = _DECISION_EFFECT
    auto_apply: bool = False
    guaranteed_outcome: bool = False
    golden_validation_performed: bool = False
    golden_protocol_eligible: bool = False
    can_bypass_decision_gates: bool = False
    changes_applied: bool = False
    configuration_change_authorized: bool = False
    cli_integration_authorized: bool = False
    report_integration_authorized: bool = False
    audit_checks: tuple[str, ...] = _AUDIT_CHECKS
    caveats: tuple[str, ...] = _SAFETY_CAVEATS
    _analysis_json: bytes = field(repr=False, compare=False)
    _seal: bytes = field(repr=False, compare=False)
    _marker: object = field(repr=False, compare=False)

    def to_public_dict(self) -> dict[str, object]:
        """Return a new, finite JSON tree reconstructed from audited bytes."""

        try:
            snapshot = _capture_result(self)
            _validate_result_snapshot(snapshot)
        except SafetyValidationError:
            raise
        except Exception:
            _fail("safety_projection_invariant_failed")
        analysis_json = snapshot["_analysis_json"]
        if type(analysis_json) is not bytes:
            _fail("safety_projection_invariant_failed")
        analysis = _strict_json_object(analysis_json)
        analysis_status = snapshot["analysis_status"]
        current = snapshot["current_max_num_seqs"]
        candidate = snapshot["candidate_max_num_seqs"]
        concurrency = snapshot["offered_concurrency"]
        handoff: dict[str, object] | None
        if current is None:
            handoff = None
        else:
            handoff = {
                "field": "max_num_seqs",
                "baseline_value": current,
                "candidate_value": candidate,
                "closed_loop_concurrency": concurrency,
                "pair_representable": True,
                "golden_validation_performed": False,
                "golden_protocol_eligible": False,
                "scheduler_saturation_proven": False,
                "required_next_step": _GOLDEN_NEXT_STEP,
            }
        return {
            "schema_version": SAFETY_VALIDATION_SCHEMA_VERSION,
            "artifact_type": SAFETY_VALIDATION_ARTIFACT_TYPE,
            "scope": SAFETY_VALIDATION_SCOPE,
            "status": "passed_safety_boundary",
            "analysis_status": analysis_status,
            "safety_validated": True,
            "supplementary_content_validated": True,
            "decision_eligible": False,
            "decision_effect": _DECISION_EFFECT,
            "auto_apply": False,
            "guaranteed_outcome": False,
            "golden_validation_performed": False,
            "golden_protocol_eligible": False,
            "can_bypass_decision_gates": False,
            "changes_applied": False,
            "configuration_change_authorized": False,
            "cli_integration_authorized": False,
            "report_integration_authorized": False,
            "golden_handoff": handoff,
            "analysis": analysis,
            "audit_checks": list(_AUDIT_CHECKS),
            "caveats": list(_SAFETY_CAVEATS),
        }


def _make_result(
    analysis_projection: dict[str, object],
    *,
    current: int | None,
    candidate: int | None,
    concurrency: int | None,
) -> ValidatedAgentOutputs:
    analysis_json = _canonical_json(analysis_projection)
    status = analysis_projection["status"]
    if type(status) is not str:
        _fail("safety_projection_invariant_failed")
    result = object.__new__(ValidatedAgentOutputs)
    values: dict[str, object] = {
        "analysis_status": status,
        "current_max_num_seqs": current,
        "candidate_max_num_seqs": candidate,
        "offered_concurrency": concurrency,
        "schema_version": SAFETY_VALIDATION_SCHEMA_VERSION,
        "artifact_type": SAFETY_VALIDATION_ARTIFACT_TYPE,
        "scope": SAFETY_VALIDATION_SCOPE,
        "safety_validated": True,
        "supplementary_content_validated": True,
        "decision_eligible": False,
        "decision_effect": _DECISION_EFFECT,
        "auto_apply": False,
        "guaranteed_outcome": False,
        "golden_validation_performed": False,
        "golden_protocol_eligible": False,
        "can_bypass_decision_gates": False,
        "changes_applied": False,
        "configuration_change_authorized": False,
        "cli_integration_authorized": False,
        "report_integration_authorized": False,
        "audit_checks": _AUDIT_CHECKS,
        "caveats": _SAFETY_CAVEATS,
        "_analysis_json": analysis_json,
        "_seal": _seal_payload(
            analysis_json, status, current, candidate, concurrency
        ),
        "_marker": _AUDIT_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    _validate_result(result)
    return result


def _is_bounded_int(value: object, *, allow_zero: bool = False) -> bool:
    minimum = 0 if allow_zero else 1
    return type(value) is int and minimum <= value <= MAX_GOLDEN_MAX_NUM_SEQS


def _validate_context(context: BottleneckContext) -> None:
    if (
        type(context) is not BottleneckContext
        or not _is_bounded_int(context.current_max_num_seqs)
        or not _is_bounded_int(context.current_max_num_batched_tokens)
        or not _is_bounded_int(context.offered_concurrency)
        or not _is_bounded_int(context.throttle_successful_requests)
        or not _is_bounded_int(
            context.throttle_failed_requests, allow_zero=True
        )
        or type(context.effective_flags_provenance) is not str
        or context.effective_flags_provenance not in _PROVENANCE_VALUES
        or type(context.traffic_scope) is not str
        or context.traffic_scope not in _TRAFFIC_SCOPE_VALUES
    ):
        _fail("safety_invalid_context")


def _validate_window_independently(window: MetricsWindow) -> None:
    """Recheck every field that can influence the independent replay."""

    if (
        type(window.elapsed_seconds) is not float
        or not math.isfinite(window.elapsed_seconds)
        or not 0.0 < window.elapsed_seconds <= _MAX_SAFE_NUMERIC_MAGNITUDE
        or type(window.observations) is not int
        or not 2 <= window.observations <= _MAX_OBSERVATIONS + 2
        or type(window.scope) is not str
        or window.scope != "server_exporter_window"
        or type(window.decision_effect) is not str
        or window.decision_effect != _DECISION_EFFECT
        or type(window.source_families) is not tuple
        or len(window.source_families) > 64
        or any(
            type(name) is not str or not name or len(name) > 128
            for name in window.source_families
        )
        or window.source_families
        != tuple(sorted(set(window.source_families)))
        or type(window.series_counts) is not tuple
        or len(window.series_counts) > 128
        or type(window.histogram_observation_counts) is not tuple
        or len(window.histogram_observation_counts) > 16
        or (
            window.metric_scope_consistent is not None
            and type(window.metric_scope_consistent) is not bool
        )
        or (
            window.metric_scope_count is not None
            and (
                type(window.metric_scope_count) is not int
                or not 0 <= window.metric_scope_count <= 4_000
            )
        )
        or (
            (window.metric_scope_consistent is None)
            != (window.metric_scope_count is None)
        )
        or (
            window.metric_scope_consistent is True
            and window.metric_scope_count == 0
        )
    ):
        _fail("safety_invalid_metrics_window")
    integer_values = (
        window.requests_finished,
        window.output_tokens,
        window.prompt_tokens,
        window.preemptions,
        window.max_requests_running,
        window.max_requests_waiting,
        window.max_requests_swapped,
        window.allowed_finished_requests,
        window.disallowed_finished_requests,
        window.unclassified_finished_requests,
    )
    if any(
        value is not None
        and (
            type(value) is not int
            or not 0 <= value <= _MAX_SAFE_NUMERIC_MAGNITUDE
        )
        for value in integer_values
    ):
        _fail("safety_invalid_metrics_window")
    float_values = (
        window.finished_requests_per_second,
        window.output_tokens_per_second,
        window.prompt_tokens_per_second,
        window.mean_ttft_ms,
        window.mean_tpot_ms,
        window.mean_inter_token_latency_ms,
        window.mean_e2e_ms,
        window.mean_queue_time_ms,
        window.mean_prefill_time_ms,
        window.mean_decode_time_ms,
        window.max_kv_cache_usage_fraction,
    )
    if any(
        value is not None
        and (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= _MAX_SAFE_NUMERIC_MAGNITUDE
        )
        for value in float_values
    ) or (
        window.max_kv_cache_usage_fraction is not None
        and window.max_kv_cache_usage_fraction > 1.0
    ):
        _fail("safety_invalid_metrics_window")
    names: list[str] = []
    for item in window.series_counts:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or len(item[0]) > 128
            or type(item[1]) is not int
            or not 1 <= item[1] <= 4_000
        ):
            _fail("safety_invalid_metrics_window")
        names.append(item[0])
    if names != sorted(set(names)):
        _fail("safety_invalid_metrics_window")
    observation_names: list[str] = []
    for item in window.histogram_observation_counts:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] not in {
                "ttft",
                "tpot",
                "itl",
                "e2e",
                "queue",
                "prefill",
                "decode",
            }
            or type(item[1]) is not int
            or not 0 <= item[1] <= _MAX_SAFE_NUMERIC_MAGNITUDE
        ):
            _fail("safety_invalid_metrics_window")
        observation_names.append(item[0])
    if observation_names != sorted(set(observation_names)):
        _fail("safety_invalid_metrics_window")


def _validate_input_objects(
    window: MetricsWindow,
    context: BottleneckContext,
    analysis: BottleneckAnalysis,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(window) is not MetricsWindow:
        _fail("safety_invalid_metrics_window_type")
    try:
        window_projection = MetricsWindow.to_public_dict(window)
        _validate_window_independently(window)
    except Exception:
        _fail("safety_invalid_metrics_window")
    if type(context) is not BottleneckContext:
        _fail("safety_invalid_context_type")
    try:
        _validate_context(context)
    except SafetyValidationError:
        raise
    except Exception:
        _fail("safety_invalid_context")
    if type(analysis) is not BottleneckAnalysis:
        _fail("safety_invalid_analysis_type")
    try:
        _validate_hard_policy(analysis)
    except SafetyValidationError:
        raise
    except Exception:
        _fail("safety_invalid_analysis")
    try:
        analysis_projection = BottleneckAnalysis.to_public_dict(analysis)
    except (ArithmeticError, TypeError, ValueError):
        _fail("safety_invalid_analysis")
    except Exception:
        _fail("safety_invalid_analysis")
    if type(window_projection) is not dict or type(analysis_projection) is not dict:
        _fail("safety_projection_invariant_failed")
    _canonical_json(window_projection)
    _canonical_json(analysis_projection)
    return window_projection, analysis_projection


def _validate_hard_policy(analysis: BottleneckAnalysis) -> None:
    if type(analysis.decision_eligible) is not bool or analysis.decision_eligible:
        _fail("safety_decision_eligible_forbidden")
    if (
        type(analysis.decision_effect) is not str
        or analysis.decision_effect != _DECISION_EFFECT
    ):
        _fail("safety_decision_effect_forbidden")
    if type(analysis.auto_apply) is not bool or analysis.auto_apply:
        _fail("safety_auto_apply_forbidden")
    suggestion = analysis.suggestion
    if suggestion is None:
        return
    if type(suggestion) is not BottleneckSuggestion:
        _fail("safety_invalid_analysis")
    if type(suggestion.auto_apply) is not bool or suggestion.auto_apply:
        _fail("safety_auto_apply_forbidden")
    if (
        type(suggestion.guaranteed_outcome) is not bool
        or suggestion.guaranteed_outcome
    ):
        _fail("safety_guaranteed_outcome_forbidden")
    if (
        type(suggestion.decision_effect) is not str
        or suggestion.decision_effect != _DECISION_EFFECT
    ):
        _fail("safety_decision_effect_forbidden")
    if (
        type(suggestion.claim_strength) is not str
        or suggestion.claim_strength != _CLAIM_STRENGTH
    ):
        _fail("safety_claim_strength_forbidden")
    if (
        type(suggestion.label) is not str
        or suggestion.label != _SUGGESTION_LABEL
    ):
        _fail("safety_suggestion_label_forbidden")
    if (
        type(suggestion.validation_required) is not str
        or suggestion.validation_required != _VALIDATION_REQUIRED
    ):
        _fail("safety_validation_requirement_missing")
    if (
        type(suggestion.direction) is not str
        or suggestion.direction not in _FORMULAS
        or type(suggestion.hypothesis) is not str
        or suggestion.hypothesis != _HYPOTHESES[suggestion.direction]
        or type(suggestion.risk) is not str
        or suggestion.risk != _RISKS[suggestion.direction]
        or type(suggestion.formula_id) is not str
        or suggestion.formula_id != _FORMULAS[suggestion.direction]
        or type(suggestion.target_field) is not str
        or suggestion.target_field != "max_num_seqs"
    ):
        _fail("safety_non_allowlisted_text")
    if (
        type(suggestion.current_golden_support) is not str
        or suggestion.current_golden_support != _GOLDEN_SUPPORT
    ):
        _fail("safety_golden_marker_invalid")
    if (
        type(suggestion.required_next_step) is not str
        or suggestion.required_next_step != _GOLDEN_NEXT_STEP
    ):
        _fail("safety_golden_next_step_invalid")
    _validate_pair(suggestion)


def _validate_pair(suggestion: BottleneckSuggestion) -> None:
    current = suggestion.current_max_num_seqs
    candidate = suggestion.candidate_max_num_seqs
    offered = suggestion.offered_concurrency
    batched = suggestion.current_max_num_batched_tokens
    step = suggestion.policy_step
    if not all(
        _is_bounded_int(value)
        for value in (current, candidate, offered, batched, step)
    ):
        _fail("safety_golden_pair_invalid")
    if current == candidate:
        _fail("safety_golden_pair_invalid")
    if offered < max(current, candidate):
        _fail("safety_golden_load_insufficient")
    expected_step = (current + 3) // 4
    if step != expected_step:
        _fail("safety_golden_pair_invalid")
    if suggestion.direction == "increase":
        expected = min(offered, batched, current + expected_step)
        valid_direction = candidate > current
    elif suggestion.direction == "decrease":
        expected = max(1, current - expected_step)
        valid_direction = candidate < current
    else:
        _fail("safety_golden_pair_invalid")
    if candidate != expected or not valid_direction:
        _fail("safety_golden_pair_invalid")


def _required_sources_present(window: MetricsWindow) -> bool:
    sources = set(window.source_families)
    return all(
        sum(name in sources for name in group) == 1
        for group in _REQUIRED_SOURCE_GROUPS
    )


def _independent_quality_reasons(
    window: MetricsWindow, context: BottleneckContext
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if context.effective_flags_provenance != "runtime_verified":
        reasons.add("effective_flags_not_runtime_verified")
    if context.traffic_scope != "operator_attested_exclusive":
        reasons.add("traffic_scope_not_exclusive")
    if context.throttle_failed_requests != 0:
        reasons.add("throttle_requests_failed")
    if context.current_max_num_batched_tokens < context.current_max_num_seqs:
        reasons.add("max_num_batched_tokens_below_max_num_seqs")
    if window.elapsed_seconds < _MIN_ELAPSED_SECONDS:
        reasons.add("window_too_short")
    if window.observations < _MIN_OBSERVATIONS:
        reasons.add("too_few_observations")
    if window.observations >= 2:
        spacing = window.elapsed_seconds / (window.observations - 1)
        if spacing > _MAX_AVERAGE_SAMPLE_SPACING_SECONDS:
            reasons.add("sample_spacing_too_sparse")
    if (
        window.requests_finished is None
        or window.requests_finished < _MIN_FINISHED_REQUESTS
        or window.requests_finished < 2 * context.offered_concurrency
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
    required = (
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
    if any(value is None for value in required):
        reasons.add("missing_required_metric")
    if (
        window.output_tokens is not None and window.output_tokens <= 0
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
    for count, rate in (
        (window.requests_finished, window.finished_requests_per_second),
        (window.output_tokens, window.output_tokens_per_second),
        (window.prompt_tokens, window.prompt_tokens_per_second),
    ):
        if (count is None) != (rate is None):
            reasons.add("counter_rate_inconsistent")
        elif count is not None and rate is not None:
            expected = count / window.elapsed_seconds
            if rate != expected:
                reasons.add("counter_rate_inconsistent")
    if not _required_sources_present(window):
        reasons.add("required_metric_source_missing")
    if window.metric_scope_consistent is not True or window.metric_scope_count != 1:
        reasons.add("metric_scope_not_single_consistent")
    counts = dict(window.series_counts)
    if any(counts.get(name) != 1 for name in _REQUIRED_SINGLE_SERIES_KEYS):
        reasons.add("required_metric_series_count_not_one")
    running = window.max_requests_running
    waiting = window.max_requests_waiting
    if running is not None and running > context.current_max_num_seqs:
        reasons.add("running_exceeds_current_max_num_seqs")
    if (
        (running is not None and running > context.offered_concurrency)
        or (waiting is not None and waiting > context.offered_concurrency)
    ):
        reasons.add("gauge_exceeds_offered_concurrency")
    observations = dict(window.histogram_observation_counts)
    if (
        window.requests_finished is None
        or any(
            observations.get(name) != window.requests_finished
            for name in ("ttft", "e2e", "queue")
        )
    ):
        reasons.add("latency_observation_count_mismatch")
    queue = window.mean_queue_time_ms
    ttft = window.mean_ttft_ms
    e2e = window.mean_e2e_ms
    if (
        queue is not None
        and ttft is not None
        and e2e is not None
        and not (0.0 <= queue <= ttft <= e2e and e2e > 0.0)
    ):
        reasons.add("latency_metrics_inconsistent")
    if not reasons.issubset(_QUALITY_REASONS):
        _fail("safety_projection_invariant_failed")
    return tuple(sorted(reasons))


def _signals_projection(
    window: MetricsWindow, context: BottleneckContext
) -> dict[str, object]:
    values = (
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
        window.allowed_finished_requests,
        window.disallowed_finished_requests,
        window.unclassified_finished_requests,
    )
    if any(value is None for value in values):
        _fail("safety_projection_invariant_failed")
    finished = window.requests_finished
    output = window.output_tokens
    preemptions = window.preemptions
    finished_rate = window.finished_requests_per_second
    output_rate = window.output_tokens_per_second
    queue = window.mean_queue_time_ms
    ttft = window.mean_ttft_ms
    e2e = window.mean_e2e_ms
    running = window.max_requests_running
    waiting = window.max_requests_waiting
    kv = window.max_kv_cache_usage_fraction
    allowed = window.allowed_finished_requests
    disallowed = window.disallowed_finished_requests
    unclassified = window.unclassified_finished_requests
    latency_observations = dict(window.histogram_observation_counts).get("e2e")
    if latency_observations is None:
        _fail("safety_projection_invariant_failed")
    assert type(finished) is int
    assert type(output) is int
    assert type(preemptions) is int
    assert type(finished_rate) is float
    assert type(output_rate) is float
    assert type(queue) is float
    assert type(ttft) is float
    assert type(e2e) is float
    assert type(running) is int
    assert type(waiting) is int
    assert type(kv) is float
    assert type(allowed) is int
    assert type(disallowed) is int
    assert type(unclassified) is int
    return {
        "elapsed_seconds": window.elapsed_seconds,
        "observations": window.observations,
        "average_sample_spacing_seconds": (
            window.elapsed_seconds / (window.observations - 1)
        ),
        "sampled_kv_headroom_fraction": 1.0 - kv,
        "queue_share_of_mean_e2e": queue / e2e,
        "preemptions_per_100_finished": 100.0 * preemptions / finished,
        "max_requests_running": running,
        "max_requests_waiting": waiting,
        "max_kv_cache_usage_fraction": kv,
        "preemptions": preemptions,
        "mean_queue_time_ms": queue,
        "mean_ttft_ms": ttft,
        "mean_e2e_ms": e2e,
        "output_tokens": output,
        "output_tokens_per_second": output_rate,
        "finished_requests": finished,
        "finished_requests_per_second": finished_rate,
        "allowed_finished_requests": allowed,
        "disallowed_finished_requests": disallowed,
        "unclassified_finished_requests": unclassified,
        "latency_observations": latency_observations,
        "max_requests_swapped": window.max_requests_swapped,
        "current_max_num_seqs": context.current_max_num_seqs,
        "current_max_num_batched_tokens": context.current_max_num_batched_tokens,
        "offered_concurrency": context.offered_concurrency,
        "configured_limit_exercised": (
            context.offered_concurrency >= context.current_max_num_seqs
            and running == context.current_max_num_seqs
        ),
    }


def _no_suggestion_reasons(signals: dict[str, object]) -> tuple[str, ...]:
    reasons: set[str] = set()
    exercised = signals["configured_limit_exercised"]
    swapped = signals["max_requests_swapped"]
    preemptions = signals["preemptions"]
    kv = signals["max_kv_cache_usage_fraction"]
    waiting = signals["max_requests_waiting"]
    queue_share = signals["queue_share_of_mean_e2e"]
    offered = signals["offered_concurrency"]
    current = signals["current_max_num_seqs"]
    assert type(exercised) is bool
    assert swapped is None or type(swapped) is int
    assert type(preemptions) is int
    assert type(kv) is float
    assert type(waiting) is int
    assert type(queue_share) is float
    assert type(offered) is int
    assert type(current) is int
    if not exercised:
        reasons.add("configured_limit_not_exercised")
    if swapped is not None and swapped > 0:
        reasons.add("legacy_swap_pressure_observed")
    pressure_complete = preemptions > 0 and kv >= _HIGH_KV_FRACTION
    queue_complete = (
        preemptions == 0
        and waiting > 0
        and queue_share >= _MATERIAL_QUEUE_SHARE
        and kv <= _LOW_KV_FRACTION
        and offered > current
    )
    if pressure_complete or queue_complete:
        return tuple(sorted(reasons))
    if (preemptions > 0) != (kv >= _HIGH_KV_FRACTION):
        reasons.add("pressure_signal_incomplete")
    if waiting == 0:
        reasons.add("no_material_queue")
    elif queue_share < _MATERIAL_QUEUE_SHARE:
        reasons.add("queue_share_below_policy_threshold")
    if waiting > 0 and kv > _LOW_KV_FRACTION:
        reasons.add("queue_and_kv_signals_conflict")
    if _LOW_KV_FRACTION < kv < _HIGH_KV_FRACTION:
        reasons.add("kv_headroom_not_clear")
    if not reasons:
        reasons.add("kv_headroom_not_clear")
    if not reasons.issubset(_NO_SUGGESTION_REASONS):
        _fail("safety_projection_invariant_failed")
    return tuple(sorted(reasons))


def _suggestion_projection(
    direction: str, signals: dict[str, object]
) -> dict[str, object] | None:
    current = signals["current_max_num_seqs"]
    offered = signals["offered_concurrency"]
    batched = signals["current_max_num_batched_tokens"]
    assert type(current) is int
    assert type(offered) is int
    assert type(batched) is int
    step = (current + 3) // 4
    if direction == "increase":
        candidate = min(offered, batched, current + step)
        if candidate <= current:
            return None
    else:
        candidate = max(1, current - step)
        if candidate >= current:
            return None
    return {
        "label": _SUGGESTION_LABEL,
        "target_field": "max_num_seqs",
        "direction": direction,
        "current_value": current,
        "candidate_test_value": candidate,
        "formula_id": _FORMULAS[direction],
        "math": {
            "policy_step_numerator": 1,
            "policy_step_denominator": 4,
            "policy_step": step,
            "offered_concurrency_ceiling": offered,
            "max_num_batched_tokens_ceiling": batched,
        },
        "hypothesis": _HYPOTHESES[direction],
        "risk": _RISKS[direction],
        "claim_strength": _CLAIM_STRENGTH,
        "validation_required": _VALIDATION_REQUIRED,
        "current_golden_support": _GOLDEN_SUPPORT,
        "required_next_step": _GOLDEN_NEXT_STEP,
        "decision_effect": _DECISION_EFFECT,
        "auto_apply": False,
        "guaranteed_outcome": False,
    }


def _independent_analysis_projection(
    window: MetricsWindow, context: BottleneckContext
) -> dict[str, object]:
    quality = _independent_quality_reasons(window, context)
    status: str
    no_suggestion: tuple[str, ...]
    signals: dict[str, object] | None
    suggestion: dict[str, object] | None
    if quality:
        status = "insufficient_evidence"
        no_suggestion = ()
        signals = None
        suggestion = None
    else:
        signals = _signals_projection(window, context)
        no_suggestion = _no_suggestion_reasons(signals)
        suggestion = None
        if not no_suggestion:
            direction = "decrease" if signals["preemptions"] > 0 else "increase"
            suggestion = _suggestion_projection(direction, signals)
            if suggestion is None:
                reason = (
                    "no_lower_candidate"
                    if direction == "decrease"
                    else "no_higher_candidate_within_verified_bounds"
                )
                no_suggestion = (reason,)
        status = "suggestion_available" if suggestion is not None else "no_clear_signal"
    if status not in _STATUSES:
        _fail("safety_projection_invariant_failed")
    return {
        "schema_version": _ANALYSIS_SCHEMA_VERSION,
        "artifact_type": _ANALYSIS_ARTIFACT_TYPE,
        "scope": _ANALYSIS_SCOPE,
        "status": status,
        "decision_eligible": False,
        "decision_effect": _DECISION_EFFECT,
        "auto_apply": False,
        "quality_reasons": list(quality),
        "no_suggestion_reasons": list(no_suggestion),
        "signals": signals,
        "suggestion": suggestion,
        "caveats": list(_ANALYSIS_CAVEATS),
    }


def _reason_list(value: object, allowed: frozenset[str]) -> tuple[str, ...]:
    if (
        type(value) is not list
        or len(value) > len(allowed)
        or any(type(item) is not str or item not in allowed for item in value)
        or value != sorted(set(value))
    ):
        _fail("safety_projection_invariant_failed")
    return tuple(value)


def _is_count(value: object, *, positive: bool = False) -> bool:
    minimum = 1 if positive else 0
    return (
        type(value) is int
        and minimum <= value <= _MAX_SAFE_NUMERIC_MAGNITUDE
    )


def _is_finite_float(
    value: object,
    *,
    positive: bool = False,
    maximum: float = float(_MAX_SAFE_NUMERIC_MAGNITUDE),
) -> bool:
    minimum = 0.0
    return (
        type(value) is float
        and math.isfinite(value)
        and minimum <= value <= maximum
        and (not positive or value > 0.0)
    )


def _validate_detached_signals(signals: object) -> dict[str, object]:
    expected_keys = {
        "elapsed_seconds",
        "observations",
        "average_sample_spacing_seconds",
        "sampled_kv_headroom_fraction",
        "queue_share_of_mean_e2e",
        "preemptions_per_100_finished",
        "max_requests_running",
        "max_requests_waiting",
        "max_kv_cache_usage_fraction",
        "preemptions",
        "mean_queue_time_ms",
        "mean_ttft_ms",
        "mean_e2e_ms",
        "output_tokens",
        "output_tokens_per_second",
        "finished_requests",
        "finished_requests_per_second",
        "allowed_finished_requests",
        "disallowed_finished_requests",
        "unclassified_finished_requests",
        "latency_observations",
        "max_requests_swapped",
        "current_max_num_seqs",
        "current_max_num_batched_tokens",
        "offered_concurrency",
        "configured_limit_exercised",
    }
    if type(signals) is not dict or set(signals) != expected_keys:
        _fail("safety_projection_invariant_failed")
    positive_counts = (
        signals["observations"],
        signals["finished_requests"],
        signals["latency_observations"],
        signals["output_tokens"],
    )
    nonnegative_counts = (
        signals["max_requests_running"],
        signals["max_requests_waiting"],
        signals["preemptions"],
        signals["allowed_finished_requests"],
        signals["disallowed_finished_requests"],
        signals["unclassified_finished_requests"],
    )
    positive_floats = (
        signals["elapsed_seconds"],
        signals["average_sample_spacing_seconds"],
        signals["mean_e2e_ms"],
        signals["output_tokens_per_second"],
        signals["finished_requests_per_second"],
    )
    nonnegative_floats = (
        signals["preemptions_per_100_finished"],
        signals["mean_queue_time_ms"],
        signals["mean_ttft_ms"],
    )
    fractions = (
        signals["sampled_kv_headroom_fraction"],
        signals["queue_share_of_mean_e2e"],
        signals["max_kv_cache_usage_fraction"],
    )
    swapped = signals["max_requests_swapped"]
    if (
        any(not _is_count(value, positive=True) for value in positive_counts)
        or any(
            not _is_bounded_int(value)
            for value in (
                signals["current_max_num_seqs"],
                signals["current_max_num_batched_tokens"],
                signals["offered_concurrency"],
            )
        )
        or any(not _is_count(value) for value in nonnegative_counts)
        or any(
            not _is_finite_float(value, positive=True)
            for value in positive_floats
        )
        or any(
            not _is_finite_float(value) for value in nonnegative_floats
        )
        or any(
            not _is_finite_float(value, maximum=1.0)
            for value in fractions
        )
        or (swapped is not None and not _is_count(swapped))
        or type(signals["configured_limit_exercised"]) is not bool
    ):
        _fail("safety_projection_invariant_failed")
    observations = signals["observations"]
    elapsed = signals["elapsed_seconds"]
    finished = signals["finished_requests"]
    output = signals["output_tokens"]
    current = signals["current_max_num_seqs"]
    batched = signals["current_max_num_batched_tokens"]
    offered = signals["offered_concurrency"]
    running = signals["max_requests_running"]
    waiting = signals["max_requests_waiting"]
    preemptions = signals["preemptions"]
    kv = signals["max_kv_cache_usage_fraction"]
    queue = signals["mean_queue_time_ms"]
    ttft = signals["mean_ttft_ms"]
    e2e = signals["mean_e2e_ms"]
    assert type(observations) is int
    assert type(elapsed) is float
    assert type(finished) is int
    assert type(output) is int
    assert type(current) is int
    assert type(batched) is int
    assert type(offered) is int
    assert type(running) is int
    assert type(waiting) is int
    assert type(preemptions) is int
    assert type(kv) is float
    assert type(queue) is float
    assert type(ttft) is float
    assert type(e2e) is float
    if observations < 2:
        _fail("safety_projection_invariant_failed")
    expected_spacing = elapsed / (observations - 1)
    expected_headroom = 1.0 - kv
    expected_queue_share = queue / e2e
    expected_preemption_density = 100.0 * preemptions / finished
    expected_finished_rate = finished / elapsed
    expected_output_rate = output / elapsed
    expected_exercised = offered >= current and running == current
    if (
        observations < _MIN_OBSERVATIONS
        or observations > _MAX_OBSERVATIONS + 2
        or elapsed < _MIN_ELAPSED_SECONDS
        or expected_spacing > _MAX_AVERAGE_SAMPLE_SPACING_SECONDS
        or finished < _MIN_FINISHED_REQUESTS
        or finished < 2 * offered
        or signals["latency_observations"] != finished
        or signals["allowed_finished_requests"] != finished
        or signals["disallowed_finished_requests"] != 0
        or signals["unclassified_finished_requests"] != 0
        or output < finished
        or not 0.0 <= queue <= ttft <= e2e
        or batched < current
        or running > current
        or running > offered
        or waiting > offered
        or signals["average_sample_spacing_seconds"] != expected_spacing
        or signals["sampled_kv_headroom_fraction"] != expected_headroom
        or signals["queue_share_of_mean_e2e"] != expected_queue_share
        or signals["preemptions_per_100_finished"]
        != expected_preemption_density
        or signals["finished_requests_per_second"] != expected_finished_rate
        or signals["output_tokens_per_second"] != expected_output_rate
        or signals["configured_limit_exercised"] is not expected_exercised
    ):
        _fail("safety_projection_invariant_failed")
    return signals


def _validate_detached_suggestion(
    suggestion: object, signals: dict[str, object]
) -> dict[str, object]:
    expected_keys = {
        "label",
        "target_field",
        "direction",
        "current_value",
        "candidate_test_value",
        "formula_id",
        "math",
        "hypothesis",
        "risk",
        "claim_strength",
        "validation_required",
        "current_golden_support",
        "required_next_step",
        "decision_effect",
        "auto_apply",
        "guaranteed_outcome",
    }
    if type(suggestion) is not dict or set(suggestion) != expected_keys:
        _fail("safety_projection_invariant_failed")
    direction = suggestion["direction"]
    if type(direction) is not str or direction not in _FORMULAS:
        _fail("safety_projection_invariant_failed")
    expected = _suggestion_projection(direction, signals)
    if expected is None or _canonical_json(suggestion) != _canonical_json(expected):
        _fail("safety_projection_invariant_failed")
    return suggestion


def _validate_detached_analysis(
    analysis: dict[str, object], expected_status: str
) -> None:
    expected_keys = {
        "schema_version",
        "artifact_type",
        "scope",
        "status",
        "decision_eligible",
        "decision_effect",
        "auto_apply",
        "quality_reasons",
        "no_suggestion_reasons",
        "signals",
        "suggestion",
        "caveats",
    }
    if set(analysis) != expected_keys:
        _fail("safety_projection_invariant_failed")
    status = analysis["status"]
    if (
        type(status) is not str
        or status not in _STATUSES
        or status != expected_status
        or type(analysis["schema_version"]) is not str
        or analysis["schema_version"] != _ANALYSIS_SCHEMA_VERSION
        or type(analysis["artifact_type"]) is not str
        or analysis["artifact_type"] != _ANALYSIS_ARTIFACT_TYPE
        or type(analysis["scope"]) is not str
        or analysis["scope"] != _ANALYSIS_SCOPE
        or type(analysis["decision_eligible"]) is not bool
        or analysis["decision_eligible"]
        or type(analysis["decision_effect"]) is not str
        or analysis["decision_effect"] != _DECISION_EFFECT
        or type(analysis["auto_apply"]) is not bool
        or analysis["auto_apply"]
        or type(analysis["caveats"]) is not list
        or analysis["caveats"] != list(_ANALYSIS_CAVEATS)
        or any(type(value) is not str for value in analysis["caveats"])
    ):
        _fail("safety_projection_invariant_failed")
    quality = _reason_list(analysis["quality_reasons"], _QUALITY_REASONS)
    no_suggestion = _reason_list(
        analysis["no_suggestion_reasons"], _NO_SUGGESTION_REASONS
    )
    signals_value = analysis["signals"]
    suggestion_value = analysis["suggestion"]
    if status == "insufficient_evidence":
        if (
            not quality
            or no_suggestion
            or signals_value is not None
            or suggestion_value is not None
        ):
            _fail("safety_projection_invariant_failed")
        return
    if quality:
        _fail("safety_projection_invariant_failed")
    signals = _validate_detached_signals(signals_value)
    expected_no_suggestion = _no_suggestion_reasons(signals)
    if status == "no_clear_signal":
        if (
            not no_suggestion
            or no_suggestion != expected_no_suggestion
            or suggestion_value is not None
        ):
            _fail("safety_projection_invariant_failed")
        return
    if no_suggestion or expected_no_suggestion:
        _fail("safety_projection_invariant_failed")
    _validate_detached_suggestion(suggestion_value, signals)


_RESULT_FIELD_NAMES = (
    "analysis_status",
    "current_max_num_seqs",
    "candidate_max_num_seqs",
    "offered_concurrency",
    "schema_version",
    "artifact_type",
    "scope",
    "safety_validated",
    "supplementary_content_validated",
    "decision_eligible",
    "decision_effect",
    "auto_apply",
    "guaranteed_outcome",
    "golden_validation_performed",
    "golden_protocol_eligible",
    "can_bypass_decision_gates",
    "changes_applied",
    "configuration_change_authorized",
    "cli_integration_authorized",
    "report_integration_authorized",
    "audit_checks",
    "caveats",
    "_analysis_json",
    "_seal",
    "_marker",
)


def _capture_result(result: ValidatedAgentOutputs) -> dict[str, object]:
    """Read each immutable result slot once for validation and rendering."""

    if type(result) is not ValidatedAgentOutputs:
        _fail("safety_projection_invariant_failed")
    try:
        return {
            name: object.__getattribute__(result, name)
            for name in _RESULT_FIELD_NAMES
        }
    except (AttributeError, TypeError):
        _fail("safety_projection_invariant_failed")


def _validate_result_snapshot(snapshot: dict[str, object]) -> None:
    """Validate one captured result state without reading its source again."""

    if snapshot["_marker"] is not _AUDIT_MARKER:
        _fail("safety_projection_invariant_failed")
    fixed_values = (
        (snapshot["schema_version"], SAFETY_VALIDATION_SCHEMA_VERSION),
        (snapshot["artifact_type"], SAFETY_VALIDATION_ARTIFACT_TYPE),
        (snapshot["scope"], SAFETY_VALIDATION_SCOPE),
        (snapshot["decision_effect"], _DECISION_EFFECT),
    )
    analysis_status = snapshot["analysis_status"]
    if any(
        type(actual) is not str or actual != expected
        for actual, expected in fixed_values
    ) or type(analysis_status) is not str or analysis_status not in _STATUSES:
        _fail("safety_projection_invariant_failed")
    if (
        type(snapshot["safety_validated"]) is not bool
        or not snapshot["safety_validated"]
        or type(snapshot["supplementary_content_validated"]) is not bool
        or not snapshot["supplementary_content_validated"]
        or type(snapshot["decision_eligible"]) is not bool
        or snapshot["decision_eligible"]
        or type(snapshot["auto_apply"]) is not bool
        or snapshot["auto_apply"]
        or type(snapshot["guaranteed_outcome"]) is not bool
        or snapshot["guaranteed_outcome"]
        or type(snapshot["golden_validation_performed"]) is not bool
        or snapshot["golden_validation_performed"]
        or type(snapshot["golden_protocol_eligible"]) is not bool
        or snapshot["golden_protocol_eligible"]
        or type(snapshot["can_bypass_decision_gates"]) is not bool
        or snapshot["can_bypass_decision_gates"]
        or type(snapshot["changes_applied"]) is not bool
        or snapshot["changes_applied"]
        or type(snapshot["configuration_change_authorized"]) is not bool
        or snapshot["configuration_change_authorized"]
        or type(snapshot["cli_integration_authorized"]) is not bool
        or snapshot["cli_integration_authorized"]
        or type(snapshot["report_integration_authorized"]) is not bool
        or snapshot["report_integration_authorized"]
        or type(snapshot["audit_checks"]) is not tuple
        or any(type(value) is not str for value in snapshot["audit_checks"])
        or snapshot["audit_checks"] != _AUDIT_CHECKS
        or type(snapshot["caveats"]) is not tuple
        or any(type(value) is not str for value in snapshot["caveats"])
        or snapshot["caveats"] != _SAFETY_CAVEATS
    ):
        _fail("safety_projection_invariant_failed")
    analysis_json = snapshot["_analysis_json"]
    if type(analysis_json) is not bytes:
        _fail("safety_projection_invariant_failed")
    analysis = _strict_json_object(analysis_json)
    _validate_detached_analysis(analysis, analysis_status)
    triplet = (
        snapshot["current_max_num_seqs"],
        snapshot["candidate_max_num_seqs"],
        snapshot["offered_concurrency"],
    )
    if analysis_status == "suggestion_available":
        if not all(_is_bounded_int(value) for value in triplet):
            _fail("safety_projection_invariant_failed")
        current, candidate, concurrency = triplet
        assert type(current) is int
        assert type(candidate) is int
        assert type(concurrency) is int
        if current == candidate or concurrency < max(current, candidate):
            _fail("safety_projection_invariant_failed")
        suggestion = analysis.get("suggestion")
        if type(suggestion) is not dict or (
            suggestion.get("current_value") != current
            or suggestion.get("candidate_test_value") != candidate
        ):
            _fail("safety_projection_invariant_failed")
    elif any(value is not None for value in triplet):
        _fail("safety_projection_invariant_failed")
    expected_seal = _seal_payload(
        analysis_json,
        analysis_status,
        snapshot["current_max_num_seqs"],
        snapshot["candidate_max_num_seqs"],
        snapshot["offered_concurrency"],
    )
    if type(snapshot["_seal"]) is not bytes or snapshot["_seal"] != expected_seal:
        _fail("safety_projection_invariant_failed")


def _validate_result(result: ValidatedAgentOutputs) -> None:
    try:
        _validate_result_snapshot(_capture_result(result))
    except SafetyValidationError:
        raise
    except Exception:
        _fail("safety_projection_invariant_failed")


def audit_agent_outputs(
    *,
    window: MetricsWindow,
    context: BottleneckContext,
    analysis: BottleneckAnalysis,
) -> ValidatedAgentOutputs:
    """Validate, independently replay, and detach one experimental chain.

    A successful return means only that the supplementary output obeys the
    current safety contract.  It never authorizes routing into a standard
    report or another CLI path, and never upgrades the output or a proposed
    Golden pair to decision-grade.
    """

    _, supplied = _validate_input_objects(window, context, analysis)
    expected = _independent_analysis_projection(window, context)
    if _canonical_json(supplied) != _canonical_json(expected):
        _fail("safety_analysis_input_mismatch")
    current: int | None = None
    candidate: int | None = None
    concurrency: int | None = None
    suggestion = expected["suggestion"]
    if suggestion is not None:
        if type(suggestion) is not dict:
            _fail("safety_projection_invariant_failed")
        current_value = suggestion.get("current_value")
        candidate_value = suggestion.get("candidate_test_value")
        offered_value = context.offered_concurrency
        if not all(
            _is_bounded_int(value)
            for value in (current_value, candidate_value, offered_value)
        ):
            _fail("safety_golden_pair_invalid")
        assert type(current_value) is int
        assert type(candidate_value) is int
        current = current_value
        candidate = candidate_value
        concurrency = offered_value
        if concurrency < max(current, candidate):
            _fail("safety_golden_load_insufficient")
    return _make_result(
        expected,
        current=current,
        candidate=candidate,
        concurrency=concurrency,
    )


__all__ = [
    "SAFETY_VALIDATION_ARTIFACT_TYPE",
    "SAFETY_VALIDATION_SCHEMA_VERSION",
    "SAFETY_VALIDATION_SCOPE",
    "SafetyValidationError",
    "ValidatedAgentOutputs",
    "audit_agent_outputs",
]
