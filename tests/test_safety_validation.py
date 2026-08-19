from __future__ import annotations

import ast
import inspect
import json
import random
import socket
import subprocess
import unittest
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

from throttle.bottleneck_analysis import (
    BottleneckAnalysis,
    BottleneckContext,
    analyze_bottleneck,
)
from throttle.golden import GOLDEN_POSITIONS, golden_positions
from throttle.safety_validation import (
    SafetyValidationError,
    ValidatedAgentOutputs,
    audit_agent_outputs,
)
from throttle.server_metrics import MetricsWindow


_PRIVATE_PAYLOAD = "https://user:password@private.example/secret\nheader"


def _window(**changes: object) -> MetricsWindow:
    values: dict[str, object] = {
        "elapsed_seconds": 8.0,
        "observations": 5,
        "requests_finished": 40,
        "output_tokens": 400,
        "prompt_tokens": None,
        "preemptions": 0,
        "finished_requests_per_second": 5.0,
        "output_tokens_per_second": 50.0,
        "prompt_tokens_per_second": None,
        "mean_ttft_ms": 300.0,
        "mean_tpot_ms": None,
        "mean_inter_token_latency_ms": None,
        "mean_e2e_ms": 1_000.0,
        "mean_queue_time_ms": 200.0,
        "mean_prefill_time_ms": None,
        "mean_decode_time_ms": None,
        "max_requests_running": 8,
        "max_requests_waiting": 4,
        "max_requests_swapped": 0,
        "max_kv_cache_usage_fraction": 0.70,
        "source_families": tuple(
            sorted(
                {
                    "vllm:request_success_total",
                    "vllm:generation_tokens_total",
                    "vllm:num_preemptions_total",
                    "vllm:num_requests_running",
                    "vllm:num_requests_waiting",
                    "vllm:kv_cache_usage_perc",
                    "vllm:time_to_first_token_seconds_sum",
                    "vllm:time_to_first_token_seconds_count",
                    "vllm:e2e_request_latency_seconds_sum",
                    "vllm:e2e_request_latency_seconds_count",
                    "vllm:request_queue_time_seconds_sum",
                    "vllm:request_queue_time_seconds_count",
                }
            )
        ),
        "series_counts": tuple(
            sorted(
                (name, 1)
                for name in {
                    "requests_finished",
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
        ),
        "histogram_observation_counts": (
            ("e2e", 40),
            ("queue", 40),
            ("ttft", 40),
        ),
        "metric_scope_consistent": True,
        "metric_scope_count": 1,
        "allowed_finished_requests": 40,
        "disallowed_finished_requests": 0,
        "unclassified_finished_requests": 0,
    }
    values.update(changes)
    return MetricsWindow(**values)  # type: ignore[arg-type]


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


def _pressure_window() -> MetricsWindow:
    return _window(
        preemptions=1,
        max_requests_waiting=0,
        max_kv_cache_usage_fraction=0.90,
    )


def _copy_as_subclass(value: object, subclass: type[object]) -> object:
    copied = object.__new__(subclass)
    for item in fields(value):  # type: ignore[arg-type]
        object.__setattr__(copied, item.name, getattr(value, item.name))
    return copied


class _WindowSubclass(MetricsWindow):
    pass


class _ContextSubclass(BottleneckContext):
    pass


class _AnalysisSubclass(BottleneckAnalysis):
    pass


class SafetyValidationTestCase(unittest.TestCase):
    def assert_safety_code(
        self,
        expected: str,
        callable_: object,
        *,
        secret: str = _PRIVATE_PAYLOAD,
    ) -> None:
        with self.assertRaises(SafetyValidationError) as raised:
            callable_()  # type: ignore[operator]
        self.assertEqual(raised.exception.code, expected)
        self.assertEqual(str(raised.exception), expected)
        self.assertEqual(raised.exception.args, (expected,))
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, repr(raised.exception))

    def audit(
        self,
        window: MetricsWindow | None = None,
        context: BottleneckContext | None = None,
        analysis: BottleneckAnalysis | None = None,
    ) -> ValidatedAgentOutputs:
        selected_window = _window() if window is None else window
        selected_context = _context() if context is None else context
        selected_analysis = (
            analyze_bottleneck(selected_window, selected_context)
            if analysis is None
            else analysis
        )
        return audit_agent_outputs(
            window=selected_window,
            context=selected_context,
            analysis=selected_analysis,
        )

    def assert_hard_boundary(self, result: ValidatedAgentOutputs) -> None:
        public = result.to_public_dict()
        self.assertEqual(public["status"], "passed_safety_boundary")
        self.assertIs(public["safety_validated"], True)
        self.assertIs(public["supplementary_content_validated"], True)
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
        for name in false_fields:
            with self.subTest(field=name):
                self.assertIs(public[name], False)
                self.assertIs(getattr(result, name), False)
        self.assertEqual(public["decision_effect"], "none")
        self.assertEqual(result.decision_effect, "none")


class AcceptedChainTests(SafetyValidationTestCase):
    def test_all_analysis_states_are_audited_without_upgrading_them(self) -> None:
        cases = (
            (
                "queue_8_to_10",
                _window(),
                _context(),
                "suggestion_available",
                (8, 10, 16),
            ),
            (
                "pressure_8_to_6",
                _pressure_window(),
                _context(),
                "suggestion_available",
                (8, 6, 16),
            ),
            (
                "no_signal",
                _window(max_requests_waiting=0),
                _context(),
                "no_clear_signal",
                None,
            ),
            (
                "insufficient_evidence",
                _window(),
                _context(effective_flags_provenance="unknown"),
                "insufficient_evidence",
                None,
            ),
        )
        for name, window, context, status, expected_pair in cases:
            analysis = analyze_bottleneck(window, context)
            with self.subTest(name=name):
                result = audit_agent_outputs(
                    window=window,
                    context=context,
                    analysis=analysis,
                )
                public = result.to_public_dict()
                self.assertEqual(result.analysis_status, status)
                self.assertEqual(public["analysis_status"], status)
                self.assertEqual(public["analysis"], analysis.to_public_dict())
                self.assert_hard_boundary(result)
                if expected_pair is None:
                    self.assertIsNone(public["golden_handoff"])
                    self.assertEqual(
                        (
                            result.current_max_num_seqs,
                            result.candidate_max_num_seqs,
                            result.offered_concurrency,
                        ),
                        (None, None, None),
                    )
                else:
                    current, candidate, concurrency = expected_pair
                    self.assertEqual(
                        (
                            result.current_max_num_seqs,
                            result.candidate_max_num_seqs,
                            result.offered_concurrency,
                        ),
                        expected_pair,
                    )
                    handoff = public["golden_handoff"]
                    assert isinstance(handoff, dict)
                    self.assertEqual(handoff["field"], "max_num_seqs")
                    self.assertEqual(handoff["baseline_value"], current)
                    self.assertEqual(handoff["candidate_value"], candidate)
                    self.assertEqual(
                        handoff["closed_loop_concurrency"], concurrency
                    )
                    self.assertGreaterEqual(
                        concurrency, max(current, candidate)
                    )
                    self.assertIs(handoff["pair_representable"], True)
                    self.assertIs(
                        handoff["golden_validation_performed"], False
                    )
                    self.assertIs(
                        handoff["golden_protocol_eligible"], False
                    )
                    self.assertIs(
                        handoff["scheduler_saturation_proven"], False
                    )

    def test_client_load_is_not_promoted_to_scheduler_saturation(self) -> None:
        public = self.audit().to_public_dict()
        handoff = public["golden_handoff"]
        assert isinstance(handoff, dict)
        self.assertEqual(handoff["closed_loop_concurrency"], 16)
        self.assertIs(handoff["scheduler_saturation_proven"], False)
        caveats = public["caveats"]
        assert isinstance(caveats, list)
        rendered = " ".join(caveats).lower()
        self.assertIn("offered demand", rendered)
        self.assertIn("not direct server-scheduler saturation", rendered)

    def test_absent_or_inconsistent_metric_scope_is_validly_fail_closed(
        self,
    ) -> None:
        for consistent, count in ((None, None), (False, 0)):
            window = _window(
                metric_scope_consistent=consistent,
                metric_scope_count=count,
            )
            context = _context()
            analysis = analyze_bottleneck(window, context)
            self.assertEqual(analysis.status, "insufficient_evidence")
            self.assertIn(
                "metric_scope_not_single_consistent",
                analysis.quality_reasons,
            )
            with self.subTest(consistent=consistent, count=count):
                result = audit_agent_outputs(
                    window=window,
                    context=context,
                    analysis=analysis,
                )
                self.assertEqual(
                    result.analysis_status, "insufficient_evidence"
                )
                self.assertIsNone(result.to_public_dict()["golden_handoff"])
                self.assert_hard_boundary(result)

    def test_projection_is_detached_from_later_input_tampering(self) -> None:
        window = _window()
        context = _context()
        analysis = analyze_bottleneck(window, context)
        result = audit_agent_outputs(
            window=window,
            context=context,
            analysis=analysis,
        )
        before = result.to_public_dict()
        assert analysis.suggestion is not None
        object.__setattr__(analysis.suggestion, "label", _PRIVATE_PAYLOAD)
        object.__setattr__(window, "scope", _PRIVATE_PAYLOAD)
        object.__setattr__(context, "traffic_scope", "unconfirmed")
        after = result.to_public_dict()
        self.assertEqual(after, before)
        rendered = json.dumps(after, sort_keys=True, allow_nan=False)
        self.assertNotIn(_PRIVATE_PAYLOAD, rendered)
        self.assertTrue(
            all(
                value is not window
                and value is not context
                and value is not analysis
                for value in (
                    getattr(result, item.name) for item in fields(result)
                )
            )
        )


class ReplayAndGoldenBoundaryTests(SafetyValidationTestCase):
    def test_analysis_is_exactly_bound_to_both_window_and_context(self) -> None:
        queue_window = _window()
        queue_context = _context()
        queue_analysis = analyze_bottleneck(queue_window, queue_context)
        pressure = _pressure_window()
        other_context = _context(current_max_num_seqs=9)
        cases = (
            (pressure, queue_context, queue_analysis),
            (queue_window, other_context, queue_analysis),
            (
                _window(max_requests_waiting=0),
                queue_context,
                queue_analysis,
            ),
        )
        for window, context, analysis in cases:
            with self.subTest(
                pressure=window.preemptions,
                current=context.current_max_num_seqs,
            ):
                self.assert_safety_code(
                    "safety_analysis_input_mismatch",
                    lambda window=window, context=context, analysis=analysis: (
                        audit_agent_outputs(
                            window=window,
                            context=context,
                            analysis=analysis,
                        )
                    ),
                )

        duplicate_window = _window()
        duplicate_context = _context()
        accepted = audit_agent_outputs(
            window=duplicate_window,
            context=duplicate_context,
            analysis=analyze_bottleneck(_window(), _context()),
        )
        self.assertEqual(accepted.analysis_status, "suggestion_available")

    def test_arbitrary_pair_and_historical_one_vs_eight_boundaries(self) -> None:
        expected_historical = (
            ("B1", "baseline", 1),
            ("C1", "candidate", 8),
            ("B2", "baseline", 1),
            ("C2", "candidate", 8),
            ("B3", "baseline", 1),
            ("C3", "candidate", 8),
        )
        self.assertEqual(GOLDEN_POSITIONS, expected_historical)
        self.assertEqual(golden_positions(1, 8), expected_historical)

        one_window = _window(max_requests_running=1)
        one_context = _context(
            current_max_num_seqs=1,
            offered_concurrency=8,
        )
        analysis = analyze_bottleneck(one_window, one_context)
        assert analysis.suggestion is not None
        self.assertEqual(analysis.suggestion.candidate_max_num_seqs, 2)
        result = audit_agent_outputs(
            window=one_window,
            context=one_context,
            analysis=analysis,
        )
        handoff = result.to_public_dict()["golden_handoff"]
        assert isinstance(handoff, dict)
        self.assertEqual(
            (handoff["baseline_value"], handoff["candidate_value"]),
            (1, 2),
        )
        self.assert_hard_boundary(result)

        forged = analyze_bottleneck(one_window, one_context)
        assert forged.suggestion is not None
        object.__setattr__(forged.suggestion, "candidate_max_num_seqs", 8)
        self.assert_safety_code(
            "safety_golden_pair_invalid",
            lambda: audit_agent_outputs(
                window=one_window,
                context=one_context,
                analysis=forged,
            ),
        )


class AdversarialValidationTests(SafetyValidationTestCase):
    def test_hard_policy_fields_have_specific_fail_closed_errors(self) -> None:
        cases = (
            (
                "safety_decision_eligible_forbidden",
                "analysis",
                "decision_eligible",
                True,
            ),
            (
                "safety_decision_effect_forbidden",
                "analysis",
                "decision_effect",
                "decision_grade",
            ),
            (
                "safety_auto_apply_forbidden",
                "analysis",
                "auto_apply",
                True,
            ),
            (
                "safety_auto_apply_forbidden",
                "suggestion",
                "auto_apply",
                True,
            ),
            (
                "safety_guaranteed_outcome_forbidden",
                "suggestion",
                "guaranteed_outcome",
                True,
            ),
            (
                "safety_claim_strength_forbidden",
                "suggestion",
                "claim_strength",
                "guaranteed",
            ),
            (
                "safety_suggestion_label_forbidden",
                "suggestion",
                "label",
                "apply_this_guaranteed_configuration",
            ),
            (
                "safety_validation_requirement_missing",
                "suggestion",
                "validation_required",
                "none",
            ),
            (
                "safety_non_allowlisted_text",
                "suggestion",
                "hypothesis",
                "This will improve throughput.",
            ),
            (
                "safety_golden_marker_invalid",
                "suggestion",
                "current_golden_support",
                "decision_eligible",
            ),
            (
                "safety_golden_next_step_invalid",
                "suggestion",
                "required_next_step",
                "apply_now",
            ),
            (
                "safety_golden_load_insufficient",
                "suggestion",
                "offered_concurrency",
                9,
            ),
            (
                "safety_golden_pair_invalid",
                "suggestion",
                "candidate_max_num_seqs",
                8,
            ),
        )
        for expected, target, field_name, value in cases:
            analysis = analyze_bottleneck(_window(), _context())
            selected: object = analysis
            if target == "suggestion":
                assert analysis.suggestion is not None
                selected = analysis.suggestion
            object.__setattr__(selected, field_name, value)
            with self.subTest(field=field_name, expected=expected):
                self.assert_safety_code(
                    expected,
                    lambda analysis=analysis: audit_agent_outputs(
                        window=_window(),
                        context=_context(),
                        analysis=analysis,
                    ),
                )

    def test_structural_tampering_and_uninitialized_objects_use_fixed_codes(
        self,
    ) -> None:
        invalid_window = _window()
        object.__setattr__(invalid_window, "scope", _PRIVATE_PAYLOAD)
        invalid_context = _context()
        object.__setattr__(
            invalid_context, "traffic_scope", _PRIVATE_PAYLOAD
        )
        invalid_analysis = analyze_bottleneck(_window(), _context())
        object.__setattr__(invalid_analysis, "status", _PRIVATE_PAYLOAD)
        cases = (
            (
                "safety_invalid_metrics_window",
                invalid_window,
                _context(),
                analyze_bottleneck(_window(), _context()),
            ),
            (
                "safety_invalid_context",
                _window(),
                invalid_context,
                analyze_bottleneck(_window(), _context()),
            ),
            (
                "safety_invalid_analysis",
                _window(),
                _context(),
                invalid_analysis,
            ),
            (
                "safety_invalid_context",
                _window(),
                object.__new__(BottleneckContext),
                analyze_bottleneck(_window(), _context()),
            ),
            (
                "safety_invalid_analysis",
                _window(),
                _context(),
                object.__new__(BottleneckAnalysis),
            ),
        )
        for expected, window, context, analysis in cases:
            with self.subTest(expected=expected):
                self.assert_safety_code(
                    expected,
                    lambda window=window, context=context, analysis=analysis: (
                        audit_agent_outputs(
                            window=window,  # type: ignore[arg-type]
                            context=context,  # type: ignore[arg-type]
                            analysis=analysis,  # type: ignore[arg-type]
                        )
                    ),
                )

    def test_exact_types_and_arbitrary_payloads_are_rejected(self) -> None:
        valid_analysis = analyze_bottleneck(_window(), _context())
        subclass_cases = (
            (
                "safety_invalid_metrics_window_type",
                _copy_as_subclass(_window(), _WindowSubclass),
                _context(),
                valid_analysis,
            ),
            (
                "safety_invalid_context_type",
                _window(),
                _copy_as_subclass(_context(), _ContextSubclass),
                valid_analysis,
            ),
            (
                "safety_invalid_analysis_type",
                _window(),
                _context(),
                _copy_as_subclass(valid_analysis, _AnalysisSubclass),
            ),
        )
        for expected, window, context, analysis in subclass_cases:
            with self.subTest(expected=expected):
                self.assert_safety_code(
                    expected,
                    lambda window=window, context=context, analysis=analysis: (
                        audit_agent_outputs(
                            window=window,  # type: ignore[arg-type]
                            context=context,  # type: ignore[arg-type]
                            analysis=analysis,  # type: ignore[arg-type]
                        )
                    ),
                )

        type_cases = (
            (
                "safety_invalid_metrics_window_type",
                _PRIVATE_PAYLOAD,
                _context(),
                valid_analysis,
            ),
            (
                "safety_invalid_context_type",
                _window(),
                {"secret": _PRIVATE_PAYLOAD},
                valid_analysis,
            ),
            (
                "safety_invalid_analysis_type",
                _window(),
                _context(),
                [_PRIVATE_PAYLOAD],
            ),
        )
        for expected, window, context, analysis in type_cases:
            with self.subTest(expected=expected):
                self.assert_safety_code(
                    expected,
                    lambda window=window, context=context, analysis=analysis: (
                        audit_agent_outputs(
                            window=window,  # type: ignore[arg-type]
                            context=context,  # type: ignore[arg-type]
                            analysis=analysis,  # type: ignore[arg-type]
                        )
                    ),
                )

    def test_method_shadowing_cannot_bypass_unbound_validation(self) -> None:
        window = _window()
        object.__setattr__(
            window,
            "to_public_dict",
            lambda: {
                "scope": "forged",
                "decision_effect": "decision_grade",
                "secret": _PRIVATE_PAYLOAD,
            },
        )
        analysis = analyze_bottleneck(window, _context())
        result = audit_agent_outputs(
            window=window,
            context=_context(),
            analysis=analysis,
        )
        rendered = json.dumps(
            result.to_public_dict(), sort_keys=True, allow_nan=False
        )
        self.assertNotIn(_PRIVATE_PAYLOAD, rendered)
        self.assert_hard_boundary(result)

    def test_result_tampering_and_uninitialized_result_fail_projection(self) -> None:
        mutations = (
            ("decision_eligible", True),
            ("auto_apply", True),
            ("guaranteed_outcome", True),
            ("golden_validation_performed", True),
            ("golden_protocol_eligible", True),
            ("can_bypass_decision_gates", True),
            ("changes_applied", True),
            ("configuration_change_authorized", True),
            ("cli_integration_authorized", True),
            ("report_integration_authorized", True),
            ("analysis_status", "decision_grade"),
            ("candidate_max_num_seqs", 1),
            ("_analysis_json", b'{"secret":"payload"}'),
            (
                "_analysis_json",
                b'{"status":"no_clear_signal","status":"forged"}',
            ),
            ("_seal", b"forged"),
        )
        for field_name, value in mutations:
            result = self.audit()
            object.__setattr__(result, field_name, value)
            with self.subTest(field=field_name):
                self.assert_safety_code(
                    "safety_projection_invariant_failed",
                    result.to_public_dict,
                )
        self.assert_safety_code(
            "safety_projection_invariant_failed",
            object.__new__(ValidatedAgentOutputs).to_public_dict,
        )

    def test_recomputed_seal_cannot_authorize_unsafe_detached_payload(self) -> None:
        import throttle.safety_validation as safety_module

        for extra_key in (False, True):
            result = self.audit()
            analysis = json.loads(result._analysis_json)
            analysis["decision_eligible"] = True
            analysis["auto_apply"] = True
            suggestion = analysis["suggestion"]
            assert isinstance(suggestion, dict)
            suggestion["auto_apply"] = True
            suggestion["guaranteed_outcome"] = True
            if extra_key:
                analysis["private"] = _PRIVATE_PAYLOAD
            encoded = safety_module._canonical_json(analysis)
            seal = safety_module._seal_payload(
                encoded,
                result.analysis_status,
                result.current_max_num_seqs,
                result.candidate_max_num_seqs,
                result.offered_concurrency,
            )
            object.__setattr__(result, "_analysis_json", encoded)
            object.__setattr__(result, "_seal", seal)
            with self.subTest(extra_key=extra_key):
                self.assert_safety_code(
                    "safety_projection_invariant_failed",
                    result.to_public_dict,
                )

    def test_projection_renders_the_same_snapshot_it_validated(self) -> None:
        import throttle.safety_validation as safety_module

        result = self.audit()
        expected = result.to_public_dict()
        original_validate = safety_module._validate_result_snapshot

        def mutate_source_after_validation(
            snapshot: dict[str, object],
        ) -> None:
            original_validate(snapshot)
            analysis = json.loads(result._analysis_json)
            analysis["decision_eligible"] = True
            suggestion = analysis["suggestion"]
            assert isinstance(suggestion, dict)
            suggestion["guaranteed_outcome"] = True
            suggestion["hypothesis"] = "GUARANTEED"
            encoded = safety_module._canonical_json(analysis)
            object.__setattr__(result, "decision_eligible", True)
            object.__setattr__(result, "guaranteed_outcome", True)
            object.__setattr__(result, "_analysis_json", encoded)
            object.__setattr__(
                result,
                "_seal",
                safety_module._seal_payload(
                    encoded,
                    result.analysis_status,
                    result.current_max_num_seqs,
                    result.candidate_max_num_seqs,
                    result.offered_concurrency,
                ),
            )

        with patch.object(
            safety_module,
            "_validate_result_snapshot",
            side_effect=mutate_source_after_validation,
        ):
            rendered = result.to_public_dict()

        self.assertEqual(rendered, expected)
        self.assertIs(rendered["decision_eligible"], False)
        self.assertIs(rendered["guaranteed_outcome"], False)
        analysis = rendered["analysis"]
        assert isinstance(analysis, dict)
        self.assertIs(analysis["decision_eligible"], False)
        suggestion = analysis["suggestion"]
        assert isinstance(suggestion, dict)
        self.assertIs(suggestion["guaranteed_outcome"], False)
        self.assertNotEqual(suggestion["hypothesis"], "GUARANTEED")


class SerializationAndIsolationTests(SafetyValidationTestCase):
    def test_pinned_replay_contract_matches_reviewed_upstream_constants(
        self,
    ) -> None:
        import throttle.bottleneck_analysis as analyzer_module
        import throttle.safety_validation as safety_module
        import throttle.server_metrics as metrics_module

        # The safety module owns literal copies so an analyzer edit cannot
        # silently redefine the audit. This test forces any legitimate policy
        # drift through an explicit, reviewed update on both sides.
        scalar_pairs = (
            (
                safety_module._ANALYSIS_SCHEMA_VERSION,
                analyzer_module.ANALYSIS_SCHEMA_VERSION,
            ),
            (
                safety_module._ANALYSIS_ARTIFACT_TYPE,
                analyzer_module.ANALYSIS_ARTIFACT_TYPE,
            ),
            (
                safety_module._DECISION_EFFECT,
                analyzer_module.DECISION_EFFECT,
            ),
            (
                safety_module._CLAIM_STRENGTH,
                analyzer_module.CLAIM_STRENGTH,
            ),
            (
                safety_module._SUGGESTION_LABEL,
                analyzer_module.SUGGESTION_LABEL,
            ),
            (
                safety_module._VALIDATION_REQUIRED,
                analyzer_module.VALIDATION_REQUIRED,
            ),
            (
                safety_module._MIN_OBSERVATIONS,
                analyzer_module.MIN_OBSERVATIONS,
            ),
            (
                safety_module._MIN_ELAPSED_SECONDS,
                analyzer_module.MIN_ELAPSED_SECONDS,
            ),
            (
                safety_module._MAX_AVERAGE_SAMPLE_SPACING_SECONDS,
                analyzer_module.MAX_AVERAGE_SAMPLE_SPACING_SECONDS,
            ),
            (
                safety_module._MIN_FINISHED_REQUESTS,
                analyzer_module.MIN_FINISHED_REQUESTS,
            ),
            (
                safety_module._LOW_KV_FRACTION,
                analyzer_module.LOW_KV_FRACTION,
            ),
            (
                safety_module._HIGH_KV_FRACTION,
                analyzer_module.HIGH_KV_FRACTION,
            ),
            (
                safety_module._MATERIAL_QUEUE_SHARE,
                analyzer_module.MATERIAL_QUEUE_SHARE,
            ),
            (
                safety_module.MAX_GOLDEN_MAX_NUM_SEQS,
                analyzer_module.MAX_CONTEXT_INTEGER,
            ),
            (
                safety_module._MAX_OBSERVATIONS,
                metrics_module.MAX_OBSERVATIONS,
            ),
            (
                safety_module._MAX_SAFE_NUMERIC_MAGNITUDE,
                metrics_module.MAX_SAFE_NUMERIC_MAGNITUDE,
            ),
        )
        for safety_value, upstream_value in scalar_pairs:
            self.assertEqual(safety_value, upstream_value)
        self.assertEqual(
            safety_module._QUALITY_REASONS,
            analyzer_module._QUALITY_REASONS,
        )
        self.assertEqual(
            safety_module._NO_SUGGESTION_REASONS,
            analyzer_module._NO_SUGGESTION_REASONS,
        )
        self.assertEqual(
            safety_module._FORMULAS,
            analyzer_module._FORMULAS,
        )
        self.assertEqual(
            safety_module._HYPOTHESES,
            analyzer_module._HYPOTHESES,
        )
        self.assertEqual(safety_module._RISKS, analyzer_module._RISKS)
        self.assertEqual(
            safety_module._ANALYSIS_CAVEATS,
            analyzer_module._ANALYSIS_CAVEATS,
        )

    def test_weakened_upstream_policy_cannot_redefine_safety_replay(self) -> None:
        import throttle.bottleneck_analysis as analyzer_module

        window = _window(mean_queue_time_ms=100.0)
        context = _context()
        reviewed = analyze_bottleneck(window, context)
        self.assertEqual(reviewed.status, "no_clear_signal")
        accepted = self.audit(
            window=window,
            context=context,
            analysis=reviewed,
        )
        self.assertEqual(accepted.analysis_status, "no_clear_signal")

        # Simulate a future analyzer weakening its queue threshold without a
        # corresponding independent safety review. The artifact is valid to
        # that altered producer while the patch is active, but the safety
        # boundary must replay the pinned 0.20 policy and reject it.
        with patch.object(analyzer_module, "MATERIAL_QUEUE_SHARE", 0.05):
            weakened = analyze_bottleneck(window, context)
            self.assertEqual(weakened.status, "suggestion_available")
            self.assert_safety_code(
                "safety_analysis_input_mismatch",
                lambda: audit_agent_outputs(
                    window=window,
                    context=context,
                    analysis=weakened,
                ),
            )

    def test_public_projection_is_fresh_strict_json_and_label_free(self) -> None:
        result = self.audit()
        first = result.to_public_dict()
        encoded = json.dumps(
            first,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertNotIn(_PRIVATE_PAYLOAD, encoded)
        self.assertNotIn("password@", encoded)

        audit_checks = first["audit_checks"]
        caveats = first["caveats"]
        analysis = first["analysis"]
        handoff = first["golden_handoff"]
        assert isinstance(audit_checks, list)
        assert isinstance(caveats, list)
        assert isinstance(analysis, dict)
        assert isinstance(handoff, dict)
        audit_checks.clear()
        caveats.clear()
        analysis["decision_eligible"] = True
        suggestion = analysis["suggestion"]
        assert isinstance(suggestion, dict)
        suggestion["auto_apply"] = True
        handoff["golden_protocol_eligible"] = True

        second = result.to_public_dict()
        self.assertTrue(second["audit_checks"])
        self.assertTrue(second["caveats"])
        second_analysis = second["analysis"]
        assert isinstance(second_analysis, dict)
        self.assertIs(second_analysis["decision_eligible"], False)
        second_suggestion = second_analysis["suggestion"]
        assert isinstance(second_suggestion, dict)
        self.assertIs(second_suggestion["auto_apply"], False)
        second_handoff = second["golden_handoff"]
        assert isinstance(second_handoff, dict)
        self.assertIs(second_handoff["golden_protocol_eligible"], False)

    def test_audit_is_pure_and_does_not_mutate_inputs(self) -> None:
        window = _window()
        context = _context()
        analysis = analyze_bottleneck(window, context)
        before = (
            json.dumps(window.to_public_dict(), sort_keys=True),
            tuple(getattr(context, item.name) for item in fields(context)),
            json.dumps(analysis.to_public_dict(), sort_keys=True),
        )
        with (
            patch("builtins.open") as open_mock,
            patch.object(socket, "create_connection") as connect_mock,
            patch.object(subprocess, "Popen") as popen_mock,
        ):
            result = audit_agent_outputs(
                window=window,
                context=context,
                analysis=analysis,
            )
        open_mock.assert_not_called()
        connect_mock.assert_not_called()
        popen_mock.assert_not_called()
        after = (
            json.dumps(window.to_public_dict(), sort_keys=True),
            tuple(getattr(context, item.name) for item in fields(context)),
            json.dumps(analysis.to_public_dict(), sort_keys=True),
        )
        self.assertEqual(after, before)
        self.assert_hard_boundary(result)

    def test_module_stays_isolated_from_cli_reports_and_golden(self) -> None:
        import throttle.safety_validation as safety_module

        parsed = ast.parse(inspect.getsource(safety_module))
        imported_modules = {
            node.module
            for node in ast.walk(parsed)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            name.name.rsplit(".", 1)[-1]
            for node in ast.walk(parsed)
            if isinstance(node, ast.Import)
            for name in node.names
        )
        self.assertTrue(
            {
                "bottleneck_analysis",
                "server_metrics",
            }.issubset(imported_modules)
        )
        self.assertTrue(
            imported_modules.isdisjoint(
                {
                    "cli",
                    "benchmark",
                    "compare",
                    "golden",
                    "models",
                    "provenance",
                    "report",
                }
            )
        )

    def test_frozen_result_rejects_normal_mutation(self) -> None:
        result = self.audit()
        with self.assertRaises(FrozenInstanceError):
            result.decision_eligible = True  # type: ignore[misc]


class DeterministicPropertyTests(SafetyValidationTestCase):
    def test_sixty_thousand_adversarial_inputs_fail_closed(self) -> None:
        rng = random.Random(0x5AFE7A11)
        total_cases = 60_000
        queue_window = _window()
        context = _context()
        queue_analysis = analyze_bottleneck(queue_window, context)
        pressure_window = _pressure_window()
        pressure_analysis = analyze_bottleneck(pressure_window, context)
        no_signal_window = _window(max_requests_waiting=0)
        no_signal_analysis = analyze_bottleneck(no_signal_window, context)
        weak_context = _context(effective_flags_provenance="unknown")
        weak_analysis = analyze_bottleneck(queue_window, weak_context)

        forged_policy = analyze_bottleneck(_window(), _context())
        object.__setattr__(forged_policy, "decision_eligible", True)
        forged_guarantee = analyze_bottleneck(_window(), _context())
        assert forged_guarantee.suggestion is not None
        object.__setattr__(
            forged_guarantee.suggestion, "guaranteed_outcome", True
        )
        subclass_window = _copy_as_subclass(_window(), _WindowSubclass)

        observed_codes: set[str] = set()
        observed_statuses: set[str] = set()
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789:/?&=@"
        for index in range(total_cases):
            mode = index % 10
            secret = "private-" + "".join(
                rng.choice(alphabet) for _ in range(rng.randint(8, 32))
            )
            callable_: object
            if mode == 0:
                callable_ = lambda secret=secret: audit_agent_outputs(
                    window=secret,  # type: ignore[arg-type]
                    context=context,
                    analysis=queue_analysis,
                )
            elif mode == 1:
                callable_ = lambda secret=secret: audit_agent_outputs(
                    window=queue_window,
                    context={"secret": secret},  # type: ignore[arg-type]
                    analysis=queue_analysis,
                )
            elif mode == 2:
                callable_ = lambda secret=secret: audit_agent_outputs(
                    window=queue_window,
                    context=context,
                    analysis=[secret],  # type: ignore[arg-type]
                )
            elif mode == 3:
                callable_ = lambda: audit_agent_outputs(
                    window=queue_window,
                    context=context,
                    analysis=forged_policy,
                )
            elif mode == 4:
                callable_ = lambda: audit_agent_outputs(
                    window=queue_window,
                    context=context,
                    analysis=forged_guarantee,
                )
            elif mode == 5:
                callable_ = lambda: audit_agent_outputs(
                    window=subclass_window,  # type: ignore[arg-type]
                    context=context,
                    analysis=queue_analysis,
                )
            elif mode == 6:
                callable_ = lambda: audit_agent_outputs(
                    window=pressure_window,
                    context=context,
                    analysis=queue_analysis,
                )
            else:
                choices = (
                    (queue_window, context, queue_analysis),
                    (pressure_window, context, pressure_analysis),
                    (no_signal_window, context, no_signal_analysis),
                    (queue_window, weak_context, weak_analysis),
                )
                window, selected_context, analysis = choices[index % 4]
                callable_ = (
                    lambda window=window,
                    selected_context=selected_context,
                    analysis=analysis: audit_agent_outputs(
                        window=window,
                        context=selected_context,
                        analysis=analysis,
                    )
                )
            try:
                result = callable_()  # type: ignore[operator]
            except SafetyValidationError as error:
                observed_codes.add(error.code)
                self.assertEqual(str(error), error.code)
                self.assertEqual(error.args, (error.code,))
                self.assertNotIn(secret, str(error))
                self.assertNotIn(secret, repr(error))
                continue

            public = result.to_public_dict()
            observed_statuses.add(result.analysis_status)
            json.dumps(public, sort_keys=True, allow_nan=False)
            self.assertIs(public["decision_eligible"], False)
            self.assertIs(public["auto_apply"], False)
            self.assertIs(public["guaranteed_outcome"], False)
            self.assertIs(public["changes_applied"], False)
            self.assertIs(public["can_bypass_decision_gates"], False)

        self.assertEqual(
            observed_statuses,
            {
                "suggestion_available",
                "no_clear_signal",
                "insufficient_evidence",
            },
        )
        self.assertTrue(
            {
                "safety_invalid_metrics_window_type",
                "safety_invalid_context_type",
                "safety_invalid_analysis_type",
                "safety_decision_eligible_forbidden",
                "safety_guaranteed_outcome_forbidden",
                "safety_analysis_input_mismatch",
            }.issubset(observed_codes)
        )


if __name__ == "__main__":
    unittest.main()
