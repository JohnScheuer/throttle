"""Opt-in orchestration for the suggestion-only experimental agent chain.

This module is a presentation coordinator, not a new decision path.  It runs
one non-decision-grade native smoke workload while polling the explicitly
configured vLLM metrics exporter.  Only the detached projection returned by
``ValidatedAgentOutputs.to_public_dict`` may leave the coordinator.

The safety artifact deliberately keeps ``cli_integration_authorized`` and
``report_integration_authorized`` false.  Those fields mean that the artifact
cannot authorize its own routing into another CLI or report path.  The sole
reviewed presentation path is the explicit ``experimental-tuning`` command.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from .benchmark import (
    ARTIFACT_TYPE,
    MIN_DECISION_BLOCKS,
    MIN_DECISION_REQUESTS,
    MIN_DECISION_SECONDS,
    SCHEMA_VERSION,
    RunProgress,
    _best_tested as _build_best_tested,
    _manifest as _build_standard_manifest,
)
from .bottleneck_analysis import (
    MAX_AVERAGE_SAMPLE_SPACING_SECONDS,
    BottleneckContext,
    analyze_bottleneck,
)
from .compare import validate_report_structure
from .models import LoadCondition, RunConfig
from .safety_validation import (
    ValidatedAgentOutputs,
    audit_agent_outputs,
)
from .server_metrics import (
    MAX_OBSERVATIONS,
    MAX_WINDOW_RECOGNIZED_SAMPLES,
    MetricsCollectionError,
    MetricsSnapshot,
    PrometheusMetricsCollector,
    derive_metrics_window,
)
from .statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MAX_BOOTSTRAP_SAMPLE_SIZE,
)


POLL_INTERVAL_SECONDS = 1.0
MAX_ENGINE_LIMIT = 2_147_483_647
EXPERIMENTAL_ENVELOPE_SCHEMA_VERSION = "1.0"
EXPERIMENTAL_ENVELOPE_ARTIFACT_TYPE = (
    "throttle_experimental_tuning_envelope"
)

_CANONICAL_POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
_ANALYSIS_STATUSES = frozenset(
    {"insufficient_evidence", "no_clear_signal", "suggestion_available"}
)
_TRAFFIC_SCOPES = frozenset(
    {"unconfirmed", "operator_attested_exclusive"}
)
_ERROR_CODES = frozenset(
    {
        "experimental_analysis_failed",
        "experimental_engine_flag_value_invalid",
        "experimental_envelope_validation_failed",
        "experimental_internal_failure",
        "experimental_invalid_traffic_scope",
        "experimental_metrics_collection_failed",
        "experimental_metrics_observation_limit",
        "experimental_metrics_sample_gap_exceeded",
        "experimental_metrics_snapshot_invalid",
        "experimental_metrics_url_rejected",
        "experimental_metrics_window_too_large",
        "experimental_metrics_window_invalid",
        "experimental_requires_engine_flag_max_num_batched_tokens",
        "experimental_requires_engine_flag_max_num_seqs",
        "experimental_requires_native_backend",
        "experimental_requires_one_block",
        "experimental_requires_single_closed_loop_condition",
        "experimental_run_report_invalid",
        "experimental_safety_projection_failed",
        "experimental_safety_validation_failed",
        "experimental_traffic_failed",
    }
)

# Keep one reviewed callable independent of later class-attribute replacement.
# The dynamic method is still invoked and must produce exactly the same tree;
# this reference supplies the independently reconstructed expected projection.
_TRUSTED_TO_PUBLIC_DICT = ValidatedAgentOutputs.to_public_dict
_TRUSTED_SNAPSHOT_PUBLIC_SUMMARY = MetricsSnapshot.public_summary

_COMPLETE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "generated_at",
        "started_at",
        "completed_at",
        "mode",
        "status",
        "decision_eligible",
        "manifest",
        "conditions",
        "best_tested",
        "run_totals",
        "cost_summary",
        "stop_reason",
        "decision_ineligible_reasons",
        "disclaimer",
    }
)
_WORKLOAD_KEYS = frozenset(
    {
        "seed",
        "measured_sha256",
        "warmup_sha256",
        "measured_prompt_count",
        "warmup_prompt_count",
        "warmup_is_separate",
        "warmup_prompts_disjoint",
        "cache_policy",
    }
)
_CONDITION_KEYS = frozenset(
    {
        "condition",
        "valid",
        "decision_grade",
        "decision_ineligible_reasons",
        "qualification_floor",
        "warmup",
        "blocks",
        "request_counts",
        "measured_wall_seconds",
        "target_offered_request_rate",
        "achieved_offered_request_rate",
        "offered_rate_relative_error",
        "open_loop_target_achieved",
        "observed_peak_in_flight",
        "metrics",
        "diagnostic_metrics",
    }
)
_BLOCK_KEYS = frozenset(
    {
        "block_index",
        "valid",
        "invalid_reasons",
        "wall_duration_seconds",
        "request_counts",
        "offered_requests",
        "target_offered_request_rate",
        "launch_window_seconds",
        "achieved_offered_request_rate",
        "offered_rate_relative_error",
        "scheduler_lag_interval_ratio_p95",
        "open_loop_target_achieved",
        "observed_peak_in_flight",
        "scheduler_lag_ms",
        "metrics",
        "diagnostic_metrics",
    }
)
_COUNT_KEYS = frozenset(
    {
        "attempted",
        "valid",
        "invalid",
        "status_counts",
        "error_counts",
        "finish_reason_counts",
    }
)
_BASE_METRIC_KEYS = frozenset(
    {
        "valid_response_count",
        "completion_tokens",
        "prompt_tokens",
        "requests_per_second",
        "output_tokens_per_second",
        "error_rate",
        "e2e_latency_ms",
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "inter_chunk_latency_ms",
        "slo_goodput",
    }
)
_AGGREGATE_METRIC_KEYS = _BASE_METRIC_KEYS | frozenset(
    {
        "block_mean_output_tokens_per_second",
        "block_mean_output_tokens_per_second_ci",
        "block_mean_requests_per_second",
        "block_mean_requests_per_second_ci",
        "independent_ci_unit",
        "cost_per_million_output_tokens",
        "cost_metric_basis",
    }
)
_RUN_TOTAL_KEYS = frozenset(
    {
        "requests_started",
        "requests_completed",
        "requests_cancelled",
        "requests_in_flight",
        "errors",
        "reserved_output_tokens",
        "peak_in_flight",
        "elapsed_seconds",
        "cache_enabled",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate",
    }
)
_COST_SUMMARY_KEYS = frozenset(
    {
        "kind",
        "total_cost",
        "basis",
        "completion_tokens",
        "cost_per_million_output_tokens",
    }
)
_DISTRIBUTION_KEYS = frozenset(
    {"count", "mean", "p50", "p90", "p95", "p99", "p95_ci"}
)
_BOOTSTRAP_CI_KEYS = frozenset(
    {
        "low",
        "high",
        "confidence",
        "method",
        "n",
        "analysis_sample_n",
        "resamples",
    }
)
_REPEATED_BLOCK_CI_KEYS = frozenset(
    {"low", "high", "confidence", "method", "n"}
)
_ITL_KEYS = frozenset(
    {
        "count",
        "mean",
        "p50",
        "p95",
        "source",
        "unavailable_reason",
    }
)
_INTER_CHUNK_KEYS = _DISTRIBUTION_KEYS | frozenset(
    {
        "analysis_sample_count",
        "analysis_sampling",
        "source",
        "not_equivalent_to_itl",
    }
)
_SLO_GOODPUT_KEYS = frozenset(
    {
        "passing_requests",
        "denominator_valid_requests",
        "requests_per_second",
        "e2e_threshold_ms",
        "ttft_threshold_ms",
    }
)
_DISCLAIMER = (
    "Measurements describe only this exact workload and declared manifest. "
    "They are not a universal optimization, production recommendation, or "
    "savings claim."
)


class ExperimentalTuningError(ValueError):
    """A fixed, non-reflective failure from the experimental coordinator."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERROR_CODES else "experimental_internal_failure"
        super().__init__(self.code)


class MetricsCollector(Protocol):
    """The bounded collector surface used by the coordinator."""

    async def snapshot(self) -> MetricsSnapshot:
        """Return one validated metrics snapshot."""


RunTraffic = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ExperimentalTuningOutcome:
    """The ordinary smoke report and the independently audited projection."""

    report: dict[str, Any]
    validated_outputs: ValidatedAgentOutputs
    analysis_status: str

    @property
    def supplementary(self) -> dict[str, object]:
        """Revalidate and render a fresh detached projection."""

        return validated_supplementary(self)


def _fail(code: str) -> None:
    raise ExperimentalTuningError(code)


def prepare_metrics_collector(metrics_url: str) -> PrometheusMetricsCollector:
    """Validate an explicit exporter URL without performing network I/O."""

    try:
        return PrometheusMetricsCollector(metrics_url)
    except MetricsCollectionError:
        _fail("experimental_metrics_url_rejected")
    except Exception:
        _fail("experimental_metrics_url_rejected")


def _normalized_flag_name(value: str) -> str:
    return value.replace("_", "-").lower()


def _engine_limit(value: object) -> int:
    if (
        type(value) is not str
        or len(value) > 10
        or _CANONICAL_POSITIVE_INTEGER.fullmatch(value) is None
    ):
        _fail("experimental_engine_flag_value_invalid")
    parsed = int(value)
    if parsed > MAX_ENGINE_LIMIT:
        _fail("experimental_engine_flag_value_invalid")
    return parsed


def validate_experimental_config(config: RunConfig) -> tuple[int, int]:
    """Return the two canonical runtime limits required by the analyzer."""

    if type(config) is not RunConfig or config.backend != "native":
        _fail("experimental_requires_native_backend")
    if config.mode != "smoke" or config.blocks != 1:
        _fail("experimental_requires_one_block")
    if (
        type(config.conditions) is not tuple
        or len(config.conditions) != 1
        or type(config.conditions[0]) is not LoadCondition
        or config.conditions[0].kind != "closed_loop"
    ):
        _fail("experimental_requires_single_closed_loop_condition")

    limits: dict[str, str] = {}
    for name, value in config.engine_flags:
        normalized = _normalized_flag_name(name)
        if normalized in {"max-num-seqs", "max-num-batched-tokens"}:
            limits[normalized] = value
    if "max-num-seqs" not in limits:
        _fail("experimental_requires_engine_flag_max_num_seqs")
    if "max-num-batched-tokens" not in limits:
        _fail("experimental_requires_engine_flag_max_num_batched_tokens")
    return (
        _engine_limit(limits["max-num-seqs"]),
        _engine_limit(limits["max-num-batched-tokens"]),
    )


def _normalized_workload(
    value: Sequence[Sequence[Mapping[str, str]]],
) -> tuple[tuple[dict[str, str], ...], ...]:
    normalized: list[tuple[dict[str, str], ...]] = []
    try:
        for messages in value:
            checked: list[dict[str, str]] = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if (
                    type(role) is not str
                    or not role.strip()
                    or type(content) is not str
                    or not content.strip()
                ):
                    _fail("experimental_run_report_invalid")
                checked.append({"role": role, "content": content})
            if not checked:
                _fail("experimental_run_report_invalid")
            normalized.append(tuple(checked))
    except ExperimentalTuningError:
        raise
    except Exception:
        _fail("experimental_run_report_invalid")
    if not normalized:
        _fail("experimental_run_report_invalid")
    return tuple(normalized)


def _expected_manifest(
    config: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    measured = _normalized_workload(prompts)
    warmups = _normalized_workload(warmup_prompts)
    try:
        return _build_standard_manifest(config, measured, warmups)
    except Exception:
        _fail("experimental_run_report_invalid")


def _exact_nonnegative_counter_map(value: object, total: int) -> bool:
    return (
        type(value) is dict
        and all(
            type(key) is str
            and type(count) is int
            and count >= 0
            for key, count in value.items()
        )
        and sum(value.values()) == total
    )


def _count_map(value: object) -> tuple[int, int, int]:
    if type(value) is not dict or set(value) != _COUNT_KEYS:
        _fail("experimental_run_report_invalid")
    attempted = value.get("attempted")
    valid = value.get("valid")
    invalid = value.get("invalid")
    if any(
        type(item) is not int or item < 0
        for item in (attempted, valid, invalid)
    ):
        _fail("experimental_run_report_invalid")
    assert type(attempted) is int
    assert type(valid) is int
    assert type(invalid) is int
    if valid + invalid != attempted:
        _fail("experimental_run_report_invalid")
    if (
        not _exact_nonnegative_counter_map(value.get("status_counts"), attempted)
        or not _exact_nonnegative_counter_map(value.get("error_counts"), invalid)
        or not _exact_nonnegative_counter_map(
            value.get("finish_reason_counts"), valid
        )
    ):
        _fail("experimental_run_report_invalid")
    return attempted, valid, invalid


def _finite_number(value: object, *, positive: bool = False) -> bool:
    if type(value) not in {int, float} or isinstance(value, bool):
        return False
    result = float(value)
    return math.isfinite(result) and (result > 0.0 if positive else result >= 0.0)


def _close(left: object, right: object) -> bool:
    if not _finite_number(left) or not _finite_number(right):
        return False
    return math.isclose(
        float(left), float(right), rel_tol=1e-6, abs_tol=1e-9
    )


def _bootstrap_ci_is_valid(value: object, sample_count: int) -> bool:
    if type(value) is not dict or set(value) != _BOOTSTRAP_CI_KEYS:
        return False
    analysis_sample_n = value.get("analysis_sample_n")
    resamples = value.get("resamples")
    low = value.get("low")
    high = value.get("high")
    if (
        value.get("method") != "bounded_percentile_bootstrap_requests"
        or value.get("confidence") != 0.95
        or value.get("n") != sample_count
        or type(analysis_sample_n) is not int
        or analysis_sample_n != min(sample_count, MAX_BOOTSTRAP_SAMPLE_SIZE)
        or type(resamples) is not int
        or resamples != DEFAULT_BOOTSTRAP_RESAMPLES
    ):
        return False
    if sample_count < 2:
        return low is None and high is None
    return (
        _finite_number(low)
        and _finite_number(high)
        and float(low) <= float(high)
    )


def _repeated_block_ci_is_valid(value: object, expected_value: object) -> bool:
    expected_n = 1 if _finite_number(expected_value, positive=True) else 0
    return (
        type(value) is dict
        and set(value) == _REPEATED_BLOCK_CI_KEYS
        and value.get("method") == "student_t_blocks"
        and value.get("confidence") == 0.95
        and value.get("n") == expected_n
        and value.get("low") is None
        and value.get("high") is None
    )


def _distribution_is_valid(
    value: object,
    *,
    expected_count: int | None = None,
    ci_sample_count: int | None = None,
    repeated_block: bool = False,
) -> bool:
    expected_keys = _DISTRIBUTION_KEYS | (
        frozenset({"p95_repeated_block_ci"})
        if repeated_block
        else frozenset()
    )
    if type(value) is not dict or set(value) != expected_keys:
        return False
    count = value.get("count")
    if (
        type(count) is not int
        or count < 0
        or (expected_count is not None and count != expected_count)
    ):
        return False
    sample_count = count if ci_sample_count is None else ci_sample_count
    if (
        type(sample_count) is not int
        or not 0 <= sample_count <= count
        or not _bootstrap_ci_is_valid(value.get("p95_ci"), sample_count)
    ):
        return False
    statistics = tuple(value.get(name) for name in ("mean", "p50", "p90", "p95", "p99"))
    if sample_count == 0:
        if any(item is not None for item in statistics):
            return False
    elif (
        any(not _finite_number(item) for item in statistics)
        or not all(
            float(left) <= float(right)
            for left, right in zip(statistics[1:], statistics[2:])
        )
    ):
        return False
    return not repeated_block or _repeated_block_ci_is_valid(
        value.get("p95_repeated_block_ci"), value.get("p95")
    )


def _inter_chunk_is_valid(value: object) -> bool:
    if type(value) is not dict or set(value) != _INTER_CHUNK_KEYS:
        return False
    count = value.get("count")
    analysis_count = value.get("analysis_sample_count")
    return (
        type(count) is int
        and count >= 0
        and type(analysis_count) is int
        and 0 <= analysis_count <= min(count, 4_096)
        and value.get("analysis_sampling")
        == "all gaps up to 4096, then deterministic reservoir sampling"
        and value.get("source") == "client_observed_nonempty_sse_chunks"
        and value.get("not_equivalent_to_itl") is True
        and _distribution_is_valid(
            {key: value[key] for key in _DISTRIBUTION_KEYS},
            expected_count=count,
            ci_sample_count=analysis_count,
        )
    )


def _slo_goodput_is_valid(
    value: object,
    config: RunConfig,
    valid_requests: int,
    wall_seconds: float,
) -> bool:
    if config.p95_slo_ms is None and config.ttft_slo_ms is None:
        return value is None
    if type(value) is not dict or set(value) != _SLO_GOODPUT_KEYS:
        return False
    passing = value.get("passing_requests")
    return (
        type(passing) is int
        and 0 <= passing <= valid_requests
        and value.get("denominator_valid_requests") == valid_requests
        and value.get("e2e_threshold_ms") == config.p95_slo_ms
        and value.get("ttft_threshold_ms") == config.ttft_slo_ms
        and _close(
            value.get("requests_per_second"), passing / wall_seconds
        )
    )


def _manifest_matches_contract(
    manifest: object,
    config: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
) -> bool:
    if type(manifest) is not dict:
        return False
    expected = _expected_manifest(config, prompts, warmup_prompts)
    if set(manifest) != set(expected):
        return False
    if any(
        manifest.get(name) != expected[name]
        for name in expected
        if name != "workload"
    ):
        return False
    workload = manifest.get("workload")
    if type(workload) is not dict or set(workload) != _WORKLOAD_KEYS:
        return False
    return workload == expected["workload"]


def _metrics_match_counts(
    metrics: object,
    valid_requests: int,
    wall_seconds: float,
    config: RunConfig,
    *,
    aggregate: bool,
    repeated_latency: bool,
) -> bool:
    expected_keys = _AGGREGATE_METRIC_KEYS if aggregate else _BASE_METRIC_KEYS
    if type(metrics) is not dict or set(metrics) != expected_keys:
        return False
    completion_tokens = metrics.get("completion_tokens")
    prompt_tokens = metrics.get("prompt_tokens")
    if not (
        metrics.get("valid_response_count") == valid_requests
        and type(completion_tokens) is int
        and completion_tokens >= valid_requests
        and completion_tokens <= valid_requests * config.max_tokens
        and type(prompt_tokens) is int
        and prompt_tokens >= 0
        and _close(
            metrics.get("requests_per_second"),
            valid_requests / wall_seconds,
        )
        and _close(
            metrics.get("output_tokens_per_second"),
            completion_tokens / wall_seconds,
        )
        and metrics.get("error_rate") == 0.0
        and _distribution_is_valid(
            metrics.get("e2e_latency_ms"),
            expected_count=valid_requests,
            repeated_block=repeated_latency,
        )
        and _distribution_is_valid(
            metrics.get("ttft_ms"),
            expected_count=valid_requests if config.stream else 0,
            repeated_block=repeated_latency,
        )
        and _distribution_is_valid(metrics.get("tpot_ms"))
        and metrics.get("itl_ms")
        == {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "source": "unavailable",
            "unavailable_reason": (
                "native SSE chunks do not prove token boundaries"
            ),
        }
        and _inter_chunk_is_valid(metrics.get("inter_chunk_latency_ms"))
        and _slo_goodput_is_valid(
            metrics.get("slo_goodput"),
            config,
            valid_requests,
            wall_seconds,
        )
        and (not aggregate or metrics.get("independent_ci_unit") == "repeated_block")
    ):
        return False
    if not aggregate:
        return True
    output_rate = metrics.get("output_tokens_per_second")
    request_rate = metrics.get("requests_per_second")
    if (
        not _close(metrics.get("block_mean_output_tokens_per_second"), output_rate)
        or not _close(metrics.get("block_mean_requests_per_second"), request_rate)
        or not _repeated_block_ci_is_valid(
            metrics.get("block_mean_output_tokens_per_second_ci"),
            output_rate,
        )
        or not _repeated_block_ci_is_valid(
            metrics.get("block_mean_requests_per_second_ci"),
            request_rate,
        )
    ):
        return False
    if config.cost.kind == "dedicated_hourly":
        expected_cost = (
            float(config.cost.total_hourly_rate) * wall_seconds / 3600.0
        )
        return (
            metrics.get("cost_metric_basis")
            == "dedicated hourly rate times measured condition wall time"
            and _close(
                metrics.get("cost_per_million_output_tokens"),
                expected_cost / completion_tokens * 1_000_000.0,
            )
        )
    return (
        metrics.get("cost_per_million_output_tokens") is None
        and metrics.get("cost_metric_basis")
        == "whole-run or active-second cost cannot be allocated to this condition"
    )


def _complete_report_contract(
    report: object,
    config: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        type(report) is not dict
        or validate_report_structure(report) is not None
        or set(report) != _COMPLETE_REPORT_KEYS
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("artifact_type") != ARTIFACT_TYPE
        or report.get("mode") != "smoke"
        or report.get("status") != "complete"
        or report.get("decision_eligible") is not False
        or report.get("stop_reason") is not None
        or report.get("disclaimer") != _DISCLAIMER
        or not _manifest_matches_contract(
            report.get("manifest"), config, prompts, warmup_prompts
        )
    ):
        _fail("experimental_run_report_invalid")
    reasons = report.get("decision_ineligible_reasons")
    if (
        type(reasons) is not list
        or any(type(reason) is not str for reason in reasons)
        or reasons != sorted(set(reasons))
        or "smoke_mode_is_not_decision_grade" not in reasons
    ):
        _fail("experimental_run_report_invalid")
    conditions = report.get("conditions")
    if type(conditions) is not list or len(conditions) != 1:
        _fail("experimental_run_report_invalid")
    condition = conditions[0]
    if (
        type(condition) is not dict
        or set(condition) != _CONDITION_KEYS
        or condition.get("valid") is not True
        or condition.get("decision_grade") is not False
        or condition.get("condition") != config.conditions[0].public_dict()
        or condition.get("target_offered_request_rate") is not None
        or condition.get("achieved_offered_request_rate") is not None
        or condition.get("offered_rate_relative_error") is not None
        or condition.get("open_loop_target_achieved") is not None
    ):
        _fail("experimental_run_report_invalid")
    qualification = condition.get("qualification_floor")
    if qualification != {
        "minimum_valid_requests": MIN_DECISION_REQUESTS,
        "or_minimum_measured_seconds": MIN_DECISION_SECONDS,
        "minimum_blocks": MIN_DECISION_BLOCKS,
    }:
        _fail("experimental_run_report_invalid")
    blocks = condition.get("blocks")
    if type(blocks) is not list or len(blocks) != config.blocks:
        _fail("experimental_run_report_invalid")
    return condition, blocks[0]


def _parse_report_time(value: object) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _best_tested_matches(
    value: object,
    condition: Mapping[str, Any],
    config: RunConfig,
) -> bool:
    if type(value) is not dict:
        return False
    try:
        expected = _build_best_tested([condition], config)
    except Exception:
        return False
    return value == expected


def _report_times_reconcile(report: Mapping[str, Any], elapsed: float) -> bool:
    generated = _parse_report_time(report.get("generated_at"))
    started = _parse_report_time(report.get("started_at"))
    completed = _parse_report_time(report.get("completed_at"))
    return (
        generated is not None
        and started is not None
        and completed is not None
        and generated <= started <= completed
        and (completed - started).total_seconds() + 5.0 >= elapsed
    )


def _report_context_counts(
    report: object,
    config: RunConfig,
    *,
    prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
) -> tuple[int, int, int, int, int]:
    """Validate the run shape and include warm-ups in context counts."""

    if prompts is None or warmup_prompts is None:
        _fail("experimental_run_report_invalid")
    condition_report, block = _complete_report_contract(
        report, config, prompts, warmup_prompts
    )
    assert type(report) is dict
    manifest = report.get("manifest")
    assert type(manifest) is dict
    engine = manifest.get("engine")
    expected_flags = {name: value for name, value in config.engine_flags}
    if (
        type(engine) is not dict
        or engine.get("backend") != "native"
        or engine.get("backend") != config.backend
        or engine.get("effective_flags_provenance")
        != config.engine_flags_provenance
        or type(engine.get("effective_flags")) is not dict
        or engine.get("effective_flags") != expected_flags
    ):
        _fail("experimental_run_report_invalid")
    effective_flags = engine["effective_flags"]
    assert type(effective_flags) is dict
    reported_limits = {
        _normalized_flag_name(name): value
        for name, value in effective_flags.items()
        if type(name) is str
        and _normalized_flag_name(name)
        in {"max-num-seqs", "max-num-batched-tokens"}
    }
    if set(reported_limits) != {
        "max-num-seqs",
        "max-num-batched-tokens",
    }:
        _fail("experimental_run_report_invalid")
    reported_max_num_seqs = _engine_limit(reported_limits["max-num-seqs"])
    reported_max_num_batched_tokens = _engine_limit(
        reported_limits["max-num-batched-tokens"]
    )
    configured_limits = validate_experimental_config(config)
    if (
        reported_max_num_seqs,
        reported_max_num_batched_tokens,
    ) != configured_limits:
        _fail("experimental_run_report_invalid")

    expected = config.conditions[0]
    condition = condition_report.get("condition")
    if (
        type(condition) is not dict
        or condition.get("id") != expected.condition_id
        or condition.get("kind") != "closed_loop"
        or type(condition.get("value")) not in {int, float}
        or isinstance(condition.get("value"), bool)
        or float(condition["value"]) != expected.value
        or type(condition.get("max_in_flight")) is not int
        or condition.get("max_in_flight") != expected.max_in_flight
    ):
        _fail("experimental_run_report_invalid")

    warmup_attempted, warmup_valid, warmup_invalid = _count_map(
        condition_report.get("warmup")
    )
    measured_attempted, measured_valid, measured_invalid = _count_map(
        condition_report.get("request_counts")
    )
    if (
        warmup_attempted != config.warmup_requests_per_condition
        or (
            config.requests_per_block is not None
            and measured_attempted
            != config.blocks * config.requests_per_block
        )
        or measured_attempted <= 0
        or warmup_invalid != 0
        or measured_invalid != 0
        or condition_report.get("decision_ineligible_reasons")
        != ["smoke_mode_is_not_decision_grade"]
    ):
        _fail("experimental_run_report_invalid")

    if (
        type(block) is not dict
        or set(block) != _BLOCK_KEYS
        or block.get("block_index") != 1
        or block.get("valid") is not True
        or block.get("invalid_reasons") != []
        or block.get("target_offered_request_rate") is not None
        or block.get("launch_window_seconds") is not None
        or block.get("achieved_offered_request_rate") is not None
        or block.get("offered_rate_relative_error") is not None
        or block.get("scheduler_lag_interval_ratio_p95") is not None
        or block.get("open_loop_target_achieved") is not True
        or not _distribution_is_valid(
            block.get("scheduler_lag_ms"), expected_count=0
        )
    ):
        _fail("experimental_run_report_invalid")
    block_wall = block.get("wall_duration_seconds")
    if not _finite_number(block_wall, positive=True):
        _fail("experimental_run_report_invalid")
    assert type(block_wall) in {int, float}
    block_attempted, block_valid, block_invalid = _count_map(
        block.get("request_counts")
    )
    block_peak = block.get("observed_peak_in_flight")
    if (
        (block_attempted, block_valid, block_invalid)
        != (measured_attempted, measured_valid, measured_invalid)
        or block.get("request_counts")
        != condition_report.get("request_counts")
        or block.get("offered_requests") != block_attempted
        or type(block_peak) is not int
        or not 0 < block_peak <= expected.max_in_flight
        or condition_report.get("observed_peak_in_flight") != block_peak
        or not _close(
            condition_report.get("measured_wall_seconds"), block_wall
        )
        or (
            config.block_duration_seconds is not None
            and float(block_wall) < config.block_duration_seconds
        )
    ):
        _fail("experimental_run_report_invalid")

    block_metrics = block.get("metrics")
    if (
        block_metrics != block.get("diagnostic_metrics")
        or not _metrics_match_counts(
            block_metrics,
            block_valid,
            float(block_wall),
            config,
            aggregate=False,
            repeated_latency=False,
        )
    ):
        _fail("experimental_run_report_invalid")
    aggregate_metrics = condition_report.get("metrics")
    diagnostic_metrics = condition_report.get("diagnostic_metrics")
    if (
        not _metrics_match_counts(
            aggregate_metrics,
            measured_valid,
            float(block_wall),
            config,
            aggregate=True,
            repeated_latency=True,
        )
        or not _metrics_match_counts(
            diagnostic_metrics,
            measured_valid,
            float(block_wall),
            config,
            aggregate=False,
            repeated_latency=True,
        )
    ):
        _fail("experimental_run_report_invalid")
    assert type(aggregate_metrics) is dict
    assert type(diagnostic_metrics) is dict
    aggregate_only = _AGGREGATE_METRIC_KEYS - _BASE_METRIC_KEYS
    if {
        key: aggregate_metrics[key]
        for key in aggregate_metrics
        if key not in aggregate_only
    } != diagnostic_metrics:
        _fail("experimental_run_report_invalid")
    for key in _BASE_METRIC_KEYS:
        if key in {"e2e_latency_ms", "ttft_ms"}:
            aggregate_distribution = diagnostic_metrics.get(key)
            if type(aggregate_distribution) is not dict:
                _fail("experimental_run_report_invalid")
            without_repeated = {
                name: item
                for name, item in aggregate_distribution.items()
                if name != "p95_repeated_block_ci"
            }
            if without_repeated != block_metrics.get(key):
                _fail("experimental_run_report_invalid")
        elif diagnostic_metrics.get(key) != block_metrics.get(key):
            _fail("experimental_run_report_invalid")

    run_totals = report.get("run_totals")
    total_attempted = warmup_attempted + measured_attempted
    if type(run_totals) is not dict or set(run_totals) != _RUN_TOTAL_KEYS:
        _fail("experimental_run_report_invalid")
    integer_total_names = _RUN_TOTAL_KEYS - frozenset(
        {"elapsed_seconds", "cache_enabled", "cache_hit_rate"}
    )
    if any(
        type(run_totals.get(name)) is not int or run_totals[name] < 0
        for name in integer_total_names
    ):
        _fail("experimental_run_report_invalid")
    # Validate cache fields
    cache_enabled = run_totals.get("cache_enabled")
    cache_hits = run_totals.get("cache_hits")
    cache_misses = run_totals.get("cache_misses")
    cache_hit_rate = run_totals.get("cache_hit_rate")
    if (
        type(cache_enabled) is not bool
        or type(cache_hits) is not int
        or cache_hits < 0
        or type(cache_misses) is not int
        or cache_misses < 0
        or not _finite_number(cache_hit_rate, positive=False)
        or cache_hit_rate < 0.0
        or cache_hit_rate > 1.0
    ):
        _fail("experimental_run_report_invalid")
    elapsed = run_totals.get("elapsed_seconds")
    if (
        run_totals.get("requests_started") != total_attempted
        or run_totals.get("requests_completed") != total_attempted
        or run_totals.get("requests_cancelled") != 0
        or run_totals.get("requests_in_flight") != 0
        or run_totals.get("errors") != 0
        or run_totals.get("reserved_output_tokens")
        != total_attempted * config.max_tokens
        or type(run_totals.get("peak_in_flight")) is not int
        or not block_peak <= run_totals["peak_in_flight"]
        <= expected.max_in_flight
        or not _finite_number(elapsed, positive=True)
        or float(elapsed) < float(block_wall)
        or float(elapsed) > config.limits.max_elapsed_seconds
        or not _report_times_reconcile(report, float(elapsed))
    ):
        _fail("experimental_run_report_invalid")

    completion_tokens = aggregate_metrics.get("completion_tokens")
    assert type(completion_tokens) is int
    cost_summary = report.get("cost_summary")
    expected_total_cost, expected_basis = config.cost.final_cost(float(elapsed))
    expected_per_million = (
        expected_total_cost / completion_tokens * 1_000_000.0
        if expected_total_cost is not None and completion_tokens > 0
        else None
    )
    if (
        type(cost_summary) is not dict
        or set(cost_summary) != _COST_SUMMARY_KEYS
        or cost_summary.get("kind") != config.cost.kind
        or cost_summary.get("basis") != expected_basis
        or cost_summary.get("completion_tokens") != completion_tokens
        or (
            expected_total_cost is None
            and cost_summary.get("total_cost") is not None
        )
        or (
            expected_total_cost is not None
            and not _close(cost_summary.get("total_cost"), expected_total_cost)
        )
        or (
            expected_per_million is None
            and cost_summary.get("cost_per_million_output_tokens") is not None
        )
        or (
            expected_per_million is not None
            and not _close(
                cost_summary.get("cost_per_million_output_tokens"),
                expected_per_million,
            )
        )
        or not _best_tested_matches(
            report.get("best_tested"), condition_report, config
        )
    ):
        _fail("experimental_run_report_invalid")

    successful = warmup_valid + measured_valid
    failed = warmup_invalid + measured_invalid
    if (
        successful <= 0
        or successful > MAX_ENGINE_LIMIT
        or failed != 0
    ):
        _fail("experimental_run_report_invalid")
    return (
        successful,
        failed,
        expected.max_in_flight,
        reported_max_num_seqs,
        reported_max_num_batched_tokens,
    )


def validate_experimental_run_report(
    report: object,
    config: RunConfig,
    *,
    prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
) -> None:
    """Fail closed unless a report is the expected ineligible smoke artifact."""

    _report_context_counts(
        report,
        config,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )


async def _snapshot(collector: MetricsCollector) -> MetricsSnapshot:
    try:
        snapshot = await collector.snapshot()
    except asyncio.CancelledError:
        raise
    except Exception:
        _fail("experimental_metrics_collection_failed")
    if type(snapshot) is not MetricsSnapshot:
        _fail("experimental_metrics_snapshot_invalid")
    try:
        _TRUSTED_SNAPSHOT_PUBLIC_SUMMARY(snapshot)
    except Exception:
        _fail("experimental_metrics_snapshot_invalid")
    if (
        type(snapshot.recognized_samples) is not int
        or not 0 <= snapshot.recognized_samples <= MAX_WINDOW_RECOGNIZED_SAMPLES
    ):
        _fail("experimental_metrics_snapshot_invalid")
    return snapshot


def _clock_value(monotonic: Callable[[], float]) -> float:
    try:
        value = monotonic()
    except Exception:
        _fail("experimental_metrics_collection_failed")
    if type(value) not in {int, float} or isinstance(value, bool):
        _fail("experimental_metrics_collection_failed")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail("experimental_metrics_collection_failed")
    return result


def _check_gap(previous: float, current: float) -> None:
    gap = current - previous
    if gap < 0.0 or gap > MAX_AVERAGE_SAMPLE_SPACING_SECONDS:
        _fail("experimental_metrics_sample_gap_exceeded")


async def _cancel_traffic(task: asyncio.Task[dict[str, Any]]) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _canonical_json_bytes(value: object, error_code: str) -> bytes:
    if validate_report_structure(value) is not None:
        _fail(error_code)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        _fail(error_code)


def _fresh_exact_object(
    candidate: object,
    expected: object,
    error_code: str,
) -> dict[str, object]:
    candidate_json = _canonical_json_bytes(candidate, error_code)
    expected_json = _canonical_json_bytes(expected, error_code)
    if candidate_json != expected_json:
        _fail(error_code)
    try:
        reconstructed = json.loads(expected_json)
    except (UnicodeDecodeError, json.JSONDecodeError):  # pragma: no cover
        _fail(error_code)
    if type(reconstructed) is not dict:
        _fail(error_code)
    return reconstructed


def _validate_projection(
    projection: object,
    expected: object,
) -> dict[str, object]:
    """Return only a fresh tree exactly matching the reviewed projection.

    Exact canonical equality supplies a recursive key/type/value allowlist.
    The expected tree comes from the import-time unbound method operating on
    the sealed result, so status/reasons/signals/suggestion math/Golden handoff
    coherence cannot be redefined by replacing the live class attribute.
    """

    checked = _fresh_exact_object(
        projection,
        expected,
        "experimental_safety_projection_failed",
    )
    analysis_projection = checked.get("analysis")
    handoff = checked.get("golden_handoff")
    false_fields = (
        "decision_eligible",
        "auto_apply",
        "guaranteed_outcome",
        "golden_validation_performed",
        "golden_protocol_eligible",
        "can_bypass_decision_gates",
        "changes_applied",
        "configuration_change_authorized",
        "cli_integration_authorized",
        "report_integration_authorized",
    )
    if (
        checked.get("status") != "passed_safety_boundary"
        or checked.get("safety_validated") is not True
        or checked.get("supplementary_content_validated") is not True
        or checked.get("decision_effect") != "none"
        or any(checked.get(name) is not False for name in false_fields)
        or type(analysis_projection) is not dict
        or analysis_projection.get("decision_eligible") is not False
        or analysis_projection.get("auto_apply") is not False
        or analysis_projection.get("decision_effect") != "none"
        or checked.get("analysis_status") not in _ANALYSIS_STATUSES
        or analysis_projection.get("status") != checked.get("analysis_status")
    ):
        _fail("experimental_safety_projection_failed")
    suggestion = analysis_projection.get("suggestion")
    if suggestion is not None and (
        type(suggestion) is not dict
        or suggestion.get("auto_apply") is not False
        or suggestion.get("guaranteed_outcome") is not False
        or suggestion.get("decision_effect") != "none"
    ):
        _fail("experimental_safety_projection_failed")
    if handoff is not None and (
        type(handoff) is not dict
        or handoff.get("golden_validation_performed") is not False
        or handoff.get("golden_protocol_eligible") is not False
        or handoff.get("scheduler_saturation_proven") is not False
    ):
        _fail("experimental_safety_projection_failed")
    return checked


def _projection_from_validated(
    validated: ValidatedAgentOutputs,
) -> dict[str, object]:
    try:
        candidate = ValidatedAgentOutputs.to_public_dict(validated)
        expected = _TRUSTED_TO_PUBLIC_DICT(validated)
    except Exception:
        _fail("experimental_safety_projection_failed")
    return _validate_projection(candidate, expected)


def _validated_outputs(
    window: object,
    context: BottleneckContext,
) -> tuple[ValidatedAgentOutputs, dict[str, object]]:
    try:
        analysis = analyze_bottleneck(window, context)  # type: ignore[arg-type]
    except Exception:
        _fail("experimental_analysis_failed")
    try:
        validated = audit_agent_outputs(
            window=window,  # type: ignore[arg-type]
            context=context,
            analysis=analysis,
        )
    except Exception:
        _fail("experimental_safety_validation_failed")
    return validated, _projection_from_validated(validated)


def validated_supplementary(
    outcome: ExperimentalTuningOutcome,
) -> dict[str, object]:
    """Revalidate the sealed result at a CLI persistence/render seam."""

    if type(outcome) is not ExperimentalTuningOutcome:
        _fail("experimental_safety_projection_failed")
    try:
        validated = object.__getattribute__(outcome, "validated_outputs")
        expected_status = object.__getattribute__(outcome, "analysis_status")
    except Exception:
        _fail("experimental_safety_projection_failed")
    checked = _projection_from_validated(validated)
    if checked.get("analysis_status") != expected_status:
        _fail("experimental_safety_projection_failed")
    return checked


def validated_experimental_report(
    outcome: ExperimentalTuningOutcome,
    config: RunConfig,
    *,
    prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
) -> dict[str, Any]:
    """Return a fresh canonical snapshot of the validated ordinary report."""

    if type(outcome) is not ExperimentalTuningOutcome:
        _fail("experimental_run_report_invalid")
    try:
        report = object.__getattribute__(outcome, "report")
    except Exception:
        _fail("experimental_run_report_invalid")
    validate_experimental_run_report(
        report,
        config,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )
    encoded = _canonical_json_bytes(report, "experimental_run_report_invalid")
    try:
        snapshot = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError):  # pragma: no cover
        _fail("experimental_run_report_invalid")
    if type(snapshot) is not dict:
        _fail("experimental_run_report_invalid")
    validate_experimental_run_report(
        snapshot,
        config,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )
    return snapshot


def _projection_matches_report_context(
    projection: Mapping[str, object],
    report: object,
    config: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
) -> bool:
    (
        successful,
        failed,
        offered_concurrency,
        current_max_num_seqs,
        current_max_num_batched_tokens,
    ) = _report_context_counts(
        report,
        config,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )
    analysis = projection.get("analysis")
    handoff = projection.get("golden_handoff")
    if type(analysis) is not dict or failed != 0:
        return False
    status = projection.get("analysis_status")
    signals = analysis.get("signals")
    suggestion = analysis.get("suggestion")
    if status == "insufficient_evidence":
        # The public insufficient-evidence shape intentionally contains no
        # context or actionable handoff to bind. Its only linkage is the
        # envelope digest; it cannot carry a tuning candidate across runs.
        return signals is None and suggestion is None and handoff is None
    if type(signals) is not dict:
        return False
    if (
        signals.get("current_max_num_seqs") != current_max_num_seqs
        or signals.get("current_max_num_batched_tokens")
        != current_max_num_batched_tokens
        or signals.get("offered_concurrency") != offered_concurrency
        or signals.get("finished_requests") != successful
        or signals.get("allowed_finished_requests") != successful
        or signals.get("disallowed_finished_requests") != 0
        or signals.get("unclassified_finished_requests") != 0
    ):
        return False
    if status == "no_clear_signal":
        return suggestion is None and handoff is None
    if status != "suggestion_available":
        return False
    if (
        type(suggestion) is not dict
        or type(handoff) is not dict
        or type(suggestion.get("math")) is not dict
    ):
        return False
    math_projection = suggestion["math"]
    assert type(math_projection) is dict
    candidate = suggestion.get("candidate_test_value")
    return (
        suggestion.get("current_value") == current_max_num_seqs
        and math_projection.get("offered_concurrency_ceiling")
        == offered_concurrency
        and math_projection.get("max_num_batched_tokens_ceiling")
        == current_max_num_batched_tokens
        and handoff.get("baseline_value") == current_max_num_seqs
        and handoff.get("candidate_value") == candidate
        and handoff.get("closed_loop_concurrency") == offered_concurrency
    )


def _expected_envelope(
    outcome: ExperimentalTuningOutcome,
    config: RunConfig,
    ordinary_report: object,
    prompts: Sequence[Sequence[Mapping[str, str]]] | None,
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]] | None,
) -> dict[str, object]:
    current_report = validated_experimental_report(
        outcome,
        config,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )
    validate_experimental_run_report(
        ordinary_report,
        config,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )
    current_json = _canonical_json_bytes(
        current_report,
        "experimental_envelope_validation_failed",
    )
    ordinary_json = _canonical_json_bytes(
        ordinary_report,
        "experimental_envelope_validation_failed",
    )
    if current_json != ordinary_json:
        _fail("experimental_envelope_validation_failed")
    projection = validated_supplementary(outcome)
    if prompts is None or warmup_prompts is None or not _projection_matches_report_context(
        projection,
        current_report,
        config,
        prompts,
        warmup_prompts,
    ):
        _fail("experimental_envelope_validation_failed")
    return {
        "schema_version": EXPERIMENTAL_ENVELOPE_SCHEMA_VERSION,
        "artifact_type": EXPERIMENTAL_ENVELOPE_ARTIFACT_TYPE,
        "ordinary_report_sha256": hashlib.sha256(ordinary_json).hexdigest(),
        "safety_projection": projection,
    }


def validate_experimental_envelope(
    envelope: object,
    outcome: ExperimentalTuningOutcome,
    config: RunConfig,
    ordinary_report: object,
    *,
    prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
) -> dict[str, object]:
    """Return a fresh envelope only when its entire bound tree is exact.

    The digest binds equality of the sanitized ordinary report; it is not a
    signature and deliberately makes identical report content linkable.
    """

    expected = _expected_envelope(
        outcome,
        config,
        ordinary_report,
        prompts,
        warmup_prompts,
    )
    return _fresh_exact_object(
        envelope,
        expected,
        "experimental_envelope_validation_failed",
    )


def validated_experimental_envelope(
    outcome: ExperimentalTuningOutcome,
    config: RunConfig,
    ordinary_report: object | None = None,
    *,
    prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]] | None = None,
) -> dict[str, object]:
    """Build and recursively validate the fixed supplementary envelope."""

    report = (
        validated_experimental_report(
            outcome,
            config,
            prompts=prompts,
            warmup_prompts=warmup_prompts,
        )
        if ordinary_report is None
        else ordinary_report
    )
    expected = _expected_envelope(
        outcome,
        config,
        report,
        prompts,
        warmup_prompts,
    )
    return validate_experimental_envelope(
        expected,
        outcome,
        config,
        report,
        prompts=prompts,
        warmup_prompts=warmup_prompts,
    )


async def run_experimental_tuning(
    config: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    collector: MetricsCollector,
    traffic_scope: str,
    progress: RunProgress,
    run_traffic: RunTraffic,
    sample_interval_seconds: float = POLL_INTERVAL_SECONDS,
    monotonic: Callable[[], float] = time.perf_counter,
) -> ExperimentalTuningOutcome:
    """Run traffic and return only the report plus detached audited output.

    A collector or sampling-cadence failure while traffic is active cancels
    that traffic before the fixed failure is returned to the CLI.
    """

    validate_experimental_config(config)
    if traffic_scope not in _TRAFFIC_SCOPES:
        _fail("experimental_invalid_traffic_scope")
    if (
        type(sample_interval_seconds) not in {int, float}
        or isinstance(sample_interval_seconds, bool)
        or not math.isfinite(float(sample_interval_seconds))
        or not 0.0 < float(sample_interval_seconds)
        <= MAX_AVERAGE_SAMPLE_SPACING_SECONDS
    ):
        _fail("experimental_metrics_sample_gap_exceeded")

    # Keep validation evidence independent from the mutable containers given
    # to the traffic runner. A stage may mutate its own copy, but cannot
    # redefine the workload against which its returned report is checked.
    evidence_prompts = _normalized_workload(prompts)
    evidence_warmup_prompts = _normalized_workload(warmup_prompts)
    traffic_prompts = _normalized_workload(evidence_prompts)
    traffic_warmup_prompts = _normalized_workload(evidence_warmup_prompts)

    window_started = _clock_value(monotonic)
    before = await _snapshot(collector)
    total_recognized_samples = before.recognized_samples
    previous_sample_started = window_started
    observations: list[MetricsSnapshot] = []

    async def invoke_traffic() -> dict[str, Any]:
        try:
            return await run_traffic(
                config,
                traffic_prompts,
                traffic_warmup_prompts,
                progress=progress,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _fail("experimental_traffic_failed")

    traffic_task = asyncio.create_task(invoke_traffic())
    try:
        while not traffic_task.done():
            await asyncio.sleep(float(sample_interval_seconds))
            if traffic_task.done():
                break
            if len(observations) >= MAX_OBSERVATIONS:
                _fail("experimental_metrics_observation_limit")
            sample_started = _clock_value(monotonic)
            _check_gap(previous_sample_started, sample_started)
            observation = await _snapshot(collector)
            if (
                total_recognized_samples + observation.recognized_samples
                > MAX_WINDOW_RECOGNIZED_SAMPLES
            ):
                _fail("experimental_metrics_window_too_large")
            total_recognized_samples += observation.recognized_samples
            observations.append(observation)
            previous_sample_started = sample_started

        report = await traffic_task
        _report_context_counts(
            report,
            config,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
        )
        final_sample_started = _clock_value(monotonic)
        _check_gap(previous_sample_started, final_sample_started)
        after = await _snapshot(collector)
        if (
            total_recognized_samples + after.recognized_samples
            > MAX_WINDOW_RECOGNIZED_SAMPLES
        ):
            _fail("experimental_metrics_window_too_large")
    except BaseException:
        await _cancel_traffic(traffic_task)
        raise

    elapsed_seconds = final_sample_started - window_started
    try:
        window = derive_metrics_window(
            before,
            after,
            elapsed_seconds,
            observations=tuple(observations),
        )
    except Exception:
        _fail("experimental_metrics_window_invalid")

    (
        successful,
        failed,
        offered_concurrency,
        current_max_num_seqs,
        current_max_num_batched_tokens,
    ) = _report_context_counts(
        report,
        config,
        prompts=evidence_prompts,
        warmup_prompts=evidence_warmup_prompts,
    )
    try:
        context = BottleneckContext(
            current_max_num_seqs=current_max_num_seqs,
            current_max_num_batched_tokens=current_max_num_batched_tokens,
            offered_concurrency=offered_concurrency,
            throttle_successful_requests=successful,
            throttle_failed_requests=failed,
            effective_flags_provenance=config.engine_flags_provenance,
            traffic_scope=traffic_scope,
        )
    except Exception:
        _fail("experimental_analysis_failed")
    validated_outputs, supplementary = _validated_outputs(window, context)
    status = supplementary["analysis_status"]
    if type(status) is not str or status not in _ANALYSIS_STATUSES:
        _fail("experimental_safety_projection_failed")
    return ExperimentalTuningOutcome(
        report=report,
        validated_outputs=validated_outputs,
        analysis_status=status,
    )


__all__ = [
    "EXPERIMENTAL_ENVELOPE_ARTIFACT_TYPE",
    "EXPERIMENTAL_ENVELOPE_SCHEMA_VERSION",
    "ExperimentalTuningError",
    "ExperimentalTuningOutcome",
    "MAX_ENGINE_LIMIT",
    "POLL_INTERVAL_SECONDS",
    "prepare_metrics_collector",
    "run_experimental_tuning",
    "validate_experimental_envelope",
    "validate_experimental_config",
    "validate_experimental_run_report",
    "validated_experimental_envelope",
    "validated_experimental_report",
    "validated_supplementary",
]
