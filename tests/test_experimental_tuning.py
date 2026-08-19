from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

import throttle.experimental_tuning as experimental_module
from throttle.benchmark import ARTIFACT_TYPE, RunProgress
from throttle.benchmark import run_native as run_native_core
from throttle.experimental_tuning import (
    ExperimentalTuningError,
    ExperimentalTuningOutcome,
    run_experimental_tuning,
    validate_experimental_envelope,
    validate_experimental_run_report,
    validated_experimental_envelope,
)
from throttle.models import EndpointConfig, LoadCondition, RunConfig
from throttle.server_metrics import PrometheusMetricsCollector


_REAL_ASYNCIO_SLEEP = asyncio.sleep
_FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "experimental-tuning-vllm-docs"
)
_PRIVATE_STAGE_PAYLOAD = (
    "https://user:password@private-stage.example/secret\nAuthorization: token"
)
_MEASURED_PROMPTS = (
    ({"role": "user", "content": "public measured prompt"},),
)
_WARMUP_PROMPTS = (
    ({"role": "user", "content": "public warmup prompt"},),
)


def _fixture_bodies() -> tuple[bytes, ...]:
    return tuple(
        (_FIXTURE_DIRECTORY / f"snapshot-{index}.prom").read_bytes()
        for index in range(5)
    )


def _checked_projection_invariants(
    projection: dict[str, object],
) -> dict[str, object]:
    analysis = projection["analysis"]
    handoff = projection["golden_handoff"]
    assert isinstance(analysis, dict)
    assert isinstance(handoff, dict)
    suggestion = analysis["suggestion"]
    assert isinstance(suggestion, dict)
    lock_names = (
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
    return {
        "schema_version": projection["schema_version"],
        "artifact_type": projection["artifact_type"],
        "status": projection["status"],
        "analysis_status": projection["analysis_status"],
        "hard_locks": {name: projection[name] for name in lock_names},
        "analysis": {
            "decision_eligible": analysis["decision_eligible"],
            "auto_apply": analysis["auto_apply"],
            "decision_effect": analysis["decision_effect"],
            "current_value": suggestion["current_value"],
            "candidate_test_value": suggestion["candidate_test_value"],
            "guaranteed_outcome": suggestion["guaranteed_outcome"],
        },
        "golden_handoff": {
            name: handoff[name]
            for name in (
                "baseline_value",
                "candidate_value",
                "closed_loop_concurrency",
                "golden_validation_performed",
                "golden_protocol_eligible",
                "scheduler_saturation_proven",
            )
        },
    }


def _config() -> RunConfig:
    return RunConfig(
        mode="smoke",
        backend="native",
        model="public-fixture-model",
        endpoint=EndpointConfig(
            url="http://127.0.0.1:8000/v1",
            api_key="private-fixture-key",
        ),
        conditions=(LoadCondition("closed_loop", 16.0, 16),),
        blocks=1,
        requests_per_block=37,
        warmup_requests_per_condition=3,
        engine_flags=(
            ("max_num_seqs", "8"),
            ("max_num_batched_tokens", "32"),
        ),
        engine_flags_provenance="runtime_verified",
        allow_unknown_cost=True,
    )


def _smoke_report() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "artifact_type": ARTIFACT_TYPE,
        "mode": "smoke",
        "status": "complete",
        "decision_eligible": False,
        "manifest": {
            "engine": {
                "backend": "native",
                "effective_flags": {
                    "max_num_seqs": "8",
                    "max_num_batched_tokens": "32",
                },
                "effective_flags_provenance": "runtime_verified",
            }
        },
        "conditions": [
            {
                "condition": {
                    "id": "closed_loop:16",
                    "kind": "closed_loop",
                    "value": 16,
                    "max_in_flight": 16,
                },
                "valid": True,
                "decision_grade": False,
                "warmup": {
                    "attempted": 3,
                    "valid": 3,
                    "invalid": 0,
                },
                "request_counts": {
                    "attempted": 37,
                    "valid": 37,
                    "invalid": 0,
                },
            }
        ],
    }


class _ControlledRun:
    def __init__(
        self,
        report: dict[str, Any] | None = None,
        *,
        release_after: int = 4,
    ) -> None:
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.report = _smoke_report() if report is None else report
        self.sleep_calls = 0
        self.release_after = release_after

    async def traffic(self, *args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        self.started.set()
        try:
            await self.release.wait()
            return self.report
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.finished.set()

    async def sampling_sleep(self, _: float) -> None:
        self.sleep_calls += 1
        if self.sleep_calls == self.release_after:
            self.release.set()
            while not self.finished.is_set():
                await _REAL_ASYNCIO_SLEEP(0)
        else:
            await _REAL_ASYNCIO_SLEEP(0)


class _SequenceCollector:
    def __init__(self, values: list[object]) -> None:
        self.values = values
        self.calls = 0

    async def snapshot(self) -> object:
        value = self.values[self.calls]
        self.calls += 1
        if isinstance(value, BaseException):
            raise value
        return value


def _streaming_completion() -> httpx.Response:
    events = (
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "public fixture output"},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        },
    )
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode("utf-8"))


async def _full_smoke_report(
    config: RunConfig | None = None,
    prompts: object = _MEASURED_PROMPTS,
    warmup_prompts: object = _WARMUP_PROMPTS,
) -> dict[str, Any]:
    async def delayed_response(_: httpx.Request) -> httpx.Response:
        await _REAL_ASYNCIO_SLEEP(0.001)
        return _streaming_completion()

    return await run_native_core(
        _config() if config is None else config,
        prompts,  # type: ignore[arg-type]
        warmup_prompts,  # type: ignore[arg-type]
        transport=httpx.MockTransport(delayed_response),
        progress=RunProgress(),
    )


class _ConnectedVllmFixture:
    """One MockTransport for real chat requests and exporter scrapes."""

    def __init__(
        self,
        *,
        contaminate_final: bool = False,
        reset_final: bool = False,
    ) -> None:
        self.contaminate_final = contaminate_final
        self.reset_final = reset_final
        self.lock = asyncio.Lock()
        self.wave_releases = tuple(asyncio.Event() for _ in range(4))
        self.traffic_finished = asyncio.Event()
        self.started_requests = 0
        self.finished_requests = 0
        self.active_requests = 0
        self.scrapes = 0
        self.requests: list[tuple[str, str]] = []
        self.http_requests: list[httpx.Request] = []
        self.sleep_calls = 0
        self.transport = httpx.MockTransport(self.handle)

    def _metrics_body(self) -> bytes:
        finished = self.finished_requests
        if self.contaminate_final and self.scrapes >= 5:
            finished += 1
        stop = 100 + finished
        generated = 1_000 + 10 * finished
        prompted = 500 + 5 * finished
        latency_count = 120 + finished
        ttft_sum = 36.0 + 0.3 * finished
        tpot_sum = 6.0 + 0.05 * finished
        itl_sum = 40.0 + 0.36 * finished
        itl_count = 1_000 + 9 * finished
        e2e_sum = 120.0 + finished
        queue_sum = 24.0 + 0.2 * finished
        prefill_sum = 12.0 + 0.1 * finished
        decode_sum = 84.0 + 0.7 * finished
        if self.reset_final and self.scrapes >= 5:
            generated = 1
        running = min(8, self.active_requests)
        waiting = min(8, max(0, self.active_requests - 8))
        label = 'engine="0",model_name="public-fixture/model"'
        success_label = (
            'engine="0",finished_reason="stop",'
            'model_name="public-fixture/model"'
        )
        length_label = (
            'engine="0",finished_reason="length",'
            'model_name="public-fixture/model"'
        )
        abort_label = (
            'engine="0",finished_reason="abort",'
            'model_name="public-fixture/model"'
        )
        lines = (
            f"vllm:request_success_total{{{success_label}}} {stop}",
            f"vllm:request_success_total{{{length_label}}} 20",
            f"vllm:request_success_total{{{abort_label}}} 0",
            f"vllm:generation_tokens_total{{{label}}} {generated}",
            f"vllm:prompt_tokens_total{{{label}}} {prompted}",
            f"vllm:num_preemptions_total{{{label}}} 0",
            f"vllm:num_requests_running{{{label}}} {running}",
            f"vllm:num_requests_waiting{{{label}}} {waiting}",
            f"vllm:kv_cache_usage_perc{{{label}}} 0.70",
            f"vllm:time_to_first_token_seconds_sum{{{label}}} {ttft_sum}",
            f"vllm:time_to_first_token_seconds_count{{{label}}} {latency_count}",
            f"vllm:request_time_per_output_token_seconds_sum{{{label}}} {tpot_sum}",
            f"vllm:request_time_per_output_token_seconds_count{{{label}}} {latency_count}",
            f"vllm:inter_token_latency_seconds_sum{{{label}}} {itl_sum}",
            f"vllm:inter_token_latency_seconds_count{{{label}}} {itl_count}",
            f"vllm:e2e_request_latency_seconds_sum{{{label}}} {e2e_sum}",
            f"vllm:e2e_request_latency_seconds_count{{{label}}} {latency_count}",
            f"vllm:request_queue_time_seconds_sum{{{label}}} {queue_sum}",
            f"vllm:request_queue_time_seconds_count{{{label}}} {latency_count}",
            f"vllm:request_prefill_time_seconds_sum{{{label}}} {prefill_sum}",
            f"vllm:request_prefill_time_seconds_count{{{label}}} {latency_count}",
            f"vllm:request_decode_time_seconds_sum{{{label}}} {decode_sum}",
            f"vllm:request_decode_time_seconds_count{{{label}}} {latency_count}",
            'unrelated_metric{credential="private-exporter-label"} 1',
        )
        return ("\n".join(lines) + "\n").encode("utf-8")

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.http_requests.append(request)
        self.requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/metrics":
            async with self.lock:
                self.scrapes += 1
                body = self._metrics_body()
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain; version=0.0.4"},
                stream=httpx.ByteStream(body),
            )
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            raise AssertionError(f"unexpected fixture request: {request.method} {request.url}")
        async with self.lock:
            self.started_requests += 1
            request_number = self.started_requests
            self.active_requests += 1
        if request_number > 40:
            raise AssertionError("experimental fixture sent more than 40 requests")
        wave = (request_number - 1) // 10
        try:
            await self.wave_releases[wave].wait()
        except asyncio.CancelledError:
            async with self.lock:
                self.active_requests -= 1
            raise
        async with self.lock:
            self.active_requests -= 1
            self.finished_requests += 1
        return _streaming_completion()

    async def sampling_sleep(self, _: float) -> None:
        self.sleep_calls += 1
        if not 1 <= self.sleep_calls <= 4:
            raise AssertionError("unexpected extra metrics sampling interval")
        wave = self.sleep_calls - 1
        self.wave_releases[wave].set()
        target_finished = self.sleep_calls * 10
        while self.finished_requests < target_finished:
            await _REAL_ASYNCIO_SLEEP(0)
        if self.sleep_calls < 4:
            target_started = min(40, target_finished + 16)
            while self.started_requests < target_started:
                await _REAL_ASYNCIO_SLEEP(0)
        else:
            while not self.traffic_finished.is_set():
                await _REAL_ASYNCIO_SLEEP(0)


def _assert_fixed_error(
    testcase: unittest.TestCase,
    expected: str,
    error: BaseException,
) -> None:
    testcase.assertIs(type(error), ExperimentalTuningError)
    testcase.assertEqual(str(error), expected)
    testcase.assertEqual(getattr(error, "code", None), expected)
    testcase.assertNotIn(_PRIVATE_STAGE_PAYLOAD, str(error))
    testcase.assertNotIn(_PRIVATE_STAGE_PAYLOAD, repr(error))


class ExperimentalTuningEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def _run_connected_fixture(
        self, fixture: _ConnectedVllmFixture | None = None
    ) -> tuple[object, _ConnectedVllmFixture]:
        selected = _ConnectedVllmFixture() if fixture is None else fixture
        collector = PrometheusMetricsCollector(
            "http://127.0.0.1:8000/metrics",
            transport=selected.transport,
        )

        async def live_traffic(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
        ) -> dict[str, Any]:
            try:
                return await run_native_core(
                    config,  # type: ignore[arg-type]
                    prompts,  # type: ignore[arg-type]
                    warmups,  # type: ignore[arg-type]
                    transport=selected.transport,
                    progress=progress,  # type: ignore[arg-type]
                )
            finally:
                selected.traffic_finished.set()

        clock = iter((0.0, 2.0, 4.0, 6.0, 8.0))
        with patch.object(
            experimental_module.asyncio,
            "sleep",
            side_effect=selected.sampling_sleep,
        ):
            outcome = await run_experimental_tuning(
                _config(),
                _MEASURED_PROMPTS,
                _WARMUP_PROMPTS,
                collector=collector,
                traffic_scope="operator_attested_exclusive",
                progress=RunProgress(),
                run_traffic=live_traffic,
                sample_interval_seconds=1.0,
                monotonic=lambda: next(clock),
            )
        return outcome, selected

    async def _run_fixture(
        self,
        bodies: tuple[bytes, ...] | None = None,
        *,
        config: RunConfig | None = None,
        traffic_scope: str = "operator_attested_exclusive",
    ) -> tuple[object, list[httpx.Request]]:
        selected_bodies = _fixture_bodies() if bodies is None else bodies
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            index = len(requests) - 1
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain; version=0.0.4"},
                stream=httpx.ByteStream(selected_bodies[index]),
            )

        collector = PrometheusMetricsCollector(
            "https://metrics.example/metrics",
            transport=httpx.MockTransport(handler),
        )
        selected_config = _config() if config is None else config
        controlled = _ControlledRun(await _full_smoke_report(selected_config))
        clock = iter((0.0, 2.0, 4.0, 6.0, 8.0))
        with patch.object(
            experimental_module.asyncio,
            "sleep",
            side_effect=controlled.sampling_sleep,
        ):
            outcome = await run_experimental_tuning(
                selected_config,
                _MEASURED_PROMPTS,
                _WARMUP_PROMPTS,
                collector=collector,
                traffic_scope=traffic_scope,
                progress=RunProgress(),
                run_traffic=controlled.traffic,
                sample_interval_seconds=1.0,
                monotonic=lambda: next(clock),
            )
        return outcome, requests

    def assert_hard_false_locks(self, projection: dict[str, object]) -> None:
        for name in (
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
        ):
            with self.subTest(level="safety", field=name):
                self.assertIs(projection[name], False)
        self.assertEqual(projection["decision_effect"], "none")

        analysis = projection["analysis"]
        self.assertIs(type(analysis), dict)
        assert isinstance(analysis, dict)
        self.assertIs(analysis["decision_eligible"], False)
        self.assertIs(analysis["auto_apply"], False)
        self.assertEqual(analysis["decision_effect"], "none")
        suggestion = analysis["suggestion"]
        self.assertIs(type(suggestion), dict)
        assert isinstance(suggestion, dict)
        self.assertIs(suggestion["auto_apply"], False)
        self.assertIs(suggestion["guaranteed_outcome"], False)
        self.assertEqual(suggestion["decision_effect"], "none")

        handoff = projection["golden_handoff"]
        self.assertIs(type(handoff), dict)
        assert isinstance(handoff, dict)
        self.assertIs(handoff["golden_validation_performed"], False)
        self.assertIs(handoff["golden_protocol_eligible"], False)
        self.assertIs(handoff["scheduler_saturation_proven"], False)

    def assert_report_analysis_reconciliation(
        self,
        outcome: object,
        *,
        expected_concurrency: int,
        expected_exporter_finished: int | None = None,
    ) -> None:
        report = outcome.report  # type: ignore[attr-defined]
        projection = outcome.supplementary  # type: ignore[attr-defined]
        condition = report["conditions"][0]
        engine = report["manifest"]["engine"]
        flags = engine["effective_flags"]
        measured = condition["request_counts"]["valid"]
        warmups = condition["warmup"]["valid"]
        expected_finished = measured + warmups

        analysis = projection["analysis"]
        assert isinstance(analysis, dict)
        signals = analysis["signals"]
        if outcome.analysis_status == "insufficient_evidence":  # type: ignore[attr-defined]
            self.assertIsNone(signals)
            self.assertIsNone(analysis["suggestion"])
            self.assertIsNone(projection["golden_handoff"])
            return
        assert isinstance(signals, dict)
        self.assertEqual(int(flags["max_num_seqs"]), 8)
        self.assertEqual(int(flags["max_num_batched_tokens"]), 32)
        self.assertEqual(
            condition["condition"]["max_in_flight"],
            expected_concurrency,
        )
        self.assertEqual(expected_finished, 40)
        self.assertEqual(signals["current_max_num_seqs"], 8)
        self.assertEqual(signals["current_max_num_batched_tokens"], 32)
        self.assertEqual(signals["offered_concurrency"], expected_concurrency)
        exporter_finished = (
            expected_finished
            if expected_exporter_finished is None
            else expected_exporter_finished
        )
        self.assertEqual(signals["finished_requests"], exporter_finished)
        self.assertEqual(
            signals["allowed_finished_requests"], exporter_finished
        )

        suggestion = analysis["suggestion"]
        handoff = projection["golden_handoff"]
        if suggestion is None:
            self.assertIsNone(handoff)
            return
        assert isinstance(suggestion, dict)
        assert isinstance(handoff, dict)
        self.assertEqual(suggestion["current_value"], 8)
        suggestion_math = suggestion["math"]
        assert isinstance(suggestion_math, dict)
        self.assertEqual(
            suggestion_math["offered_concurrency_ceiling"],
            expected_concurrency,
        )
        self.assertEqual(handoff["baseline_value"], 8)
        self.assertEqual(
            handoff["candidate_value"], suggestion["candidate_test_value"]
        )
        self.assertEqual(
            handoff["closed_loop_concurrency"], expected_concurrency
        )

    async def test_public_vllm_names_flow_through_connected_live_chain(self) -> None:
        outcome, fixture = await self._run_connected_fixture()
        projection = outcome.supplementary  # type: ignore[attr-defined]
        self.assertEqual(outcome.analysis_status, "suggestion_available")  # type: ignore[attr-defined]
        self.assertEqual(fixture.started_requests, 40)
        self.assertEqual(fixture.finished_requests, 40)
        self.assertEqual(
            fixture.requests.count(("POST", "/v1/chat/completions")), 40
        )
        self.assertEqual(fixture.requests.count(("GET", "/metrics")), 5)
        for request in (
            item for item in fixture.http_requests if item.method == "GET"
        ):
            self.assertEqual(request.method, "GET")
            self.assertEqual(
                str(request.url), "http://127.0.0.1:8000/metrics"
            )
            self.assertEqual(
                request.headers["accept"], "text/plain; version=0.0.4"
            )
            self.assertEqual(request.headers["accept-encoding"], "identity")
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("cookie", request.headers)

        self.assertEqual(projection["status"], "passed_safety_boundary")
        self.assertEqual(projection["analysis_status"], "suggestion_available")
        analysis = projection["analysis"]
        assert isinstance(analysis, dict)
        suggestion = analysis["suggestion"]
        assert isinstance(suggestion, dict)
        self.assertEqual(suggestion["current_value"], 8)
        self.assertEqual(suggestion["candidate_test_value"], 10)
        handoff = projection["golden_handoff"]
        assert isinstance(handoff, dict)
        self.assertEqual(handoff["baseline_value"], 8)
        self.assertEqual(handoff["candidate_value"], 10)
        self.assertEqual(handoff["closed_loop_concurrency"], 16)
        self.assert_hard_false_locks(projection)
        signals = analysis["signals"]
        assert isinstance(signals, dict)
        for supplementary_only in (
            "prompt_tokens",
            "mean_tpot_ms",
            "mean_inter_token_latency_ms",
            "mean_prefill_time_ms",
            "mean_decode_time_ms",
        ):
            self.assertNotIn(supplementary_only, signals)

        report = outcome.report  # type: ignore[attr-defined]
        self.assertEqual(report["mode"], "smoke")
        self.assertIs(report["decision_eligible"], False)
        self.assert_report_analysis_reconciliation(
            outcome,
            expected_concurrency=16,
        )
        self.assertTrue(
            set(report).isdisjoint(
                {
                    "analysis",
                    "analysis_status",
                    "experimental_tuning",
                    "golden_handoff",
                    "safety_validated",
                    "supplementary",
                }
            )
        )

        envelope = validated_experimental_envelope(
            outcome,  # type: ignore[arg-type]
            _config(),
            report,
            prompts=_MEASURED_PROMPTS,
            warmup_prompts=_WARMUP_PROMPTS,
        )
        self.assertEqual(
            set(envelope),
            {
                "schema_version",
                "artifact_type",
                "ordinary_report_sha256",
                "safety_projection",
            },
        )
        self.assertEqual(envelope["schema_version"], "1.0")
        self.assertEqual(
            envelope["artifact_type"],
            "throttle_experimental_tuning_envelope",
        )
        canonical_report = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(
            envelope["ordinary_report_sha256"],
            hashlib.sha256(canonical_report).hexdigest(),
        )
        self.assertEqual(envelope["safety_projection"], projection)
        self.assertEqual(
            validate_experimental_envelope(
                envelope,
                outcome,  # type: ignore[arg-type]
                _config(),
                report,
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            ),
            envelope,
        )

        mutated_report = json.loads(json.dumps(report))
        generated_at = mutated_report["generated_at"]
        assert isinstance(generated_at, str)
        mutated_report["generated_at"] = "2025" + generated_at[4:]
        with self.assertRaises(ExperimentalTuningError) as raised:
            validate_experimental_envelope(
                envelope,
                outcome,  # type: ignore[arg-type]
                _config(),
                mutated_report,
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            )
        _assert_fixed_error(
            self,
            "experimental_envelope_validation_failed",
            raised.exception,
        )

        no_queue_bodies = tuple(
            b"\n".join(
                line
                for line in body.splitlines()
                if b"vllm:request_queue_time_seconds" not in line
            )
            + b"\n"
            for body in _fixture_bodies()
        )
        other_outcome, _ = await self._run_fixture(no_queue_bodies)
        other_envelope = validated_experimental_envelope(
            other_outcome,  # type: ignore[arg-type]
            _config(),
            other_outcome.report,  # type: ignore[attr-defined]
            prompts=_MEASURED_PROMPTS,
            warmup_prompts=_WARMUP_PROMPTS,
        )
        self.assertEqual(
            validate_experimental_envelope(
                other_envelope,
                other_outcome,  # type: ignore[arg-type]
                _config(),
                other_outcome.report,  # type: ignore[attr-defined]
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            ),
            other_envelope,
        )
        with self.assertRaises(ExperimentalTuningError) as raised:
            validate_experimental_envelope(
                other_envelope,
                outcome,  # type: ignore[arg-type]
                _config(),
                report,
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            )
        _assert_fixed_error(
            self,
            "experimental_envelope_validation_failed",
            raised.exception,
        )

        c10_config = replace(
            _config(),
            conditions=(LoadCondition("closed_loop", 10.0, 10),),
        )
        c10_outcome, _ = await self._run_fixture(config=c10_config)
        self.assertEqual(c10_outcome.analysis_status, "suggestion_available")  # type: ignore[attr-defined]
        self.assert_report_analysis_reconciliation(
            c10_outcome,
            expected_concurrency=10,
        )
        c10_envelope = validated_experimental_envelope(
            c10_outcome,  # type: ignore[arg-type]
            c10_config,
            c10_outcome.report,  # type: ignore[attr-defined]
            prompts=_MEASURED_PROMPTS,
            warmup_prompts=_WARMUP_PROMPTS,
        )
        self.assertEqual(
            validate_experimental_envelope(
                c10_envelope,
                c10_outcome,  # type: ignore[arg-type]
                c10_config,
                c10_outcome.report,  # type: ignore[attr-defined]
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            ),
            c10_envelope,
        )
        stale_cross_pair = dict(envelope)
        stale_cross_pair["safety_projection"] = c10_envelope[
            "safety_projection"
        ]
        with self.assertRaises(ExperimentalTuningError) as raised:
            validate_experimental_envelope(
                stale_cross_pair,
                outcome,  # type: ignore[arg-type]
                _config(),
                report,
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            )
        _assert_fixed_error(
            self,
            "experimental_envelope_validation_failed",
            raised.exception,
        )

        cross_context_outcome = ExperimentalTuningOutcome(
            report=report,
            validated_outputs=c10_outcome.validated_outputs,  # type: ignore[attr-defined]
            analysis_status=c10_outcome.analysis_status,  # type: ignore[attr-defined]
        )
        with self.assertRaises(ExperimentalTuningError) as raised:
            validated_experimental_envelope(
                cross_context_outcome,
                _config(),
                report,
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            )
        _assert_fixed_error(
            self,
            "experimental_envelope_validation_failed",
            raised.exception,
        )

        envelope_rendered = json.dumps(envelope, sort_keys=True)
        for forbidden in (
            "public-fixture/model",
            "127.0.0.1",
            "private-fixture-key",
            "private-exporter-label",
        ):
            self.assertNotIn(forbidden, envelope_rendered)
        rendered = json.dumps(projection, sort_keys=True, allow_nan=False)
        self.assertNotIn("public-fixture/model", rendered)
        self.assertNotIn("127.0.0.1", rendered)
        self.assertNotIn("private-fixture-key", rendered)
        self.assertNotIn("private-exporter-label", rendered)
        expected = json.loads(
            (_FIXTURE_DIRECTORY / "expected-invariants.json").read_text(
                encoding="utf-8"
            )
        )
        nested_projection = envelope["safety_projection"]
        assert isinstance(nested_projection, dict)
        self.assertEqual(
            _checked_projection_invariants(nested_projection),
            expected,
        )

    async def test_checked_snapshots_parse_and_keep_diagnostics_supplementary(
        self,
    ) -> None:
        from throttle.server_metrics import (
            derive_metrics_window,
            parse_vllm_metrics,
        )

        snapshots = tuple(parse_vllm_metrics(body) for body in _fixture_bodies())
        window = derive_metrics_window(
            snapshots[0],
            snapshots[-1],
            8.0,
            observations=snapshots[1:-1],
        )
        self.assertEqual(window.requests_finished, 40)
        self.assertEqual(window.output_tokens, 400)
        self.assertEqual(window.prompt_tokens, 200)
        self.assertAlmostEqual(window.mean_ttft_ms or 0.0, 300.0)
        self.assertAlmostEqual(window.mean_tpot_ms or 0.0, 50.0)
        self.assertAlmostEqual(
            window.mean_inter_token_latency_ms or 0.0,
            40.0,
        )
        self.assertEqual(
            dict(window.histogram_observation_counts)["itl"],
            360,
        )
        self.assertAlmostEqual(window.mean_e2e_ms or 0.0, 1_000.0)
        self.assertAlmostEqual(window.mean_queue_time_ms or 0.0, 200.0)
        self.assertAlmostEqual(window.mean_prefill_time_ms or 0.0, 100.0)
        self.assertAlmostEqual(window.mean_decode_time_ms or 0.0, 700.0)
        self.assertIsNone(window.max_requests_swapped)
        rendered = json.dumps(window.to_public_dict(), sort_keys=True)
        self.assertNotIn("public-fixture/model", rendered)
        self.assertNotIn('"engine"', rendered)

    async def test_complete_report_requires_full_config_bound_native_contract(
        self,
    ) -> None:
        full_report = await _full_smoke_report()
        validate_experimental_run_report(
            full_report,
            _config(),
            prompts=_MEASURED_PROMPTS,
            warmup_prompts=_WARMUP_PROMPTS,
        )

        delete = object()
        cases: tuple[
            tuple[str, tuple[str | int, ...], object], ...
        ] = (
            ("missing_root_manifest", ("manifest",), delete),
            ("missing_root_run_totals", ("run_totals",), delete),
            ("missing_root_cost_summary", ("cost_summary",), delete),
            ("missing_root_best_tested", ("best_tested",), delete),
            ("extra_root", ("unreviewed_stage_output",), {}),
            ("missing_manifest_model", ("manifest", "model"), delete),
            ("missing_manifest_workload", ("manifest", "workload"), delete),
            ("missing_manifest_request", ("manifest", "request"), delete),
            ("missing_manifest_traffic", ("manifest", "traffic"), delete),
            ("missing_manifest_safety", ("manifest", "safety"), delete),
            (
                "model_mismatch",
                ("manifest", "model", "id"),
                "different-public-model",
            ),
            (
                "workload_hash_mismatch",
                ("manifest", "workload", "measured_sha256"),
                "0" * 64,
            ),
            (
                "request_max_tokens_mismatch",
                ("manifest", "request", "max_tokens"),
                129,
            ),
            (
                "request_stream_mismatch",
                ("manifest", "request", "stream"),
                False,
            ),
            (
                "traffic_requests_mismatch",
                ("manifest", "traffic", "requests_per_block"),
                36,
            ),
            (
                "traffic_warmups_mismatch",
                ("manifest", "traffic", "warmup_requests_per_condition"),
                2,
            ),
            (
                "traffic_condition_mismatch",
                (
                    "manifest",
                    "traffic",
                    "conditions",
                    0,
                    "max_in_flight",
                ),
                15,
            ),
            (
                "safety_limit_mismatch",
                ("manifest", "safety", "limits", "max_requests"),
                9_999,
            ),
            (
                "condition_count_mismatch",
                ("conditions", 0, "request_counts", "valid"),
                36,
            ),
            (
                "missing_block_request_counts",
                ("conditions", 0, "blocks", 0, "request_counts"),
                delete,
            ),
            (
                "missing_block_metrics",
                ("conditions", 0, "blocks", 0, "metrics"),
                delete,
            ),
            (
                "block_index_mismatch",
                ("conditions", 0, "blocks", 0, "block_index"),
                2,
            ),
            (
                "block_valid_mismatch",
                ("conditions", 0, "blocks", 0, "valid"),
                False,
            ),
            (
                "block_count_mismatch",
                ("conditions", 0, "blocks", 0, "request_counts", "valid"),
                36,
            ),
            (
                "run_total_mismatch",
                ("run_totals", "requests_completed"),
                39,
            ),
            (
                "completion_token_total_mismatch",
                ("cost_summary", "completion_tokens"),
                369,
            ),
        )

        with self.assertRaises(ExperimentalTuningError) as raised:
            validate_experimental_run_report(
                _smoke_report(),
                _config(),
                prompts=_MEASURED_PROMPTS,
                warmup_prompts=_WARMUP_PROMPTS,
            )
        _assert_fixed_error(
            self,
            "experimental_run_report_invalid",
            raised.exception,
        )

        for name, path, value in cases:
            with self.subTest(case=name):
                report = json.loads(json.dumps(full_report))
                target: object = report
                for segment in path[:-1]:
                    if isinstance(segment, int):
                        assert isinstance(target, list)
                        target = target[segment]
                    else:
                        assert isinstance(target, dict)
                        target = target[segment]
                final = path[-1]
                if isinstance(final, int):
                    assert isinstance(target, list)
                    if value is delete:
                        del target[final]
                    else:
                        target[final] = value
                else:
                    assert isinstance(target, dict)
                    if value is delete:
                        del target[final]
                    else:
                        target[final] = value
                with self.assertRaises(ExperimentalTuningError) as raised:
                    validate_experimental_run_report(
                        report,
                        _config(),
                        prompts=_MEASURED_PROMPTS,
                        warmup_prompts=_WARMUP_PROMPTS,
                    )
                _assert_fixed_error(
                    self,
                    "experimental_run_report_invalid",
                    raised.exception,
                )

    async def test_no_clear_signal_keeps_report_context_reconciled(self) -> None:
        queue_sums = (20, 21, 22, 23, 24)
        bodies: list[bytes] = []
        for body, queue_sum in zip(
            _fixture_bodies(), queue_sums, strict=True
        ):
            lines = []
            for line in body.splitlines():
                if line.startswith(b"vllm:request_queue_time_seconds_sum{"):
                    labels = line.split(b"} ", 1)[0]
                    line = labels + b"} " + str(queue_sum).encode("ascii")
                lines.append(line)
            bodies.append(b"\n".join(lines) + b"\n")

        outcome, _ = await self._run_fixture(tuple(bodies))
        self.assertEqual(outcome.analysis_status, "no_clear_signal")  # type: ignore[attr-defined]
        self.assert_report_analysis_reconciliation(
            outcome,
            expected_concurrency=16,
        )
        projection = outcome.supplementary  # type: ignore[attr-defined]
        analysis = projection["analysis"]
        assert isinstance(analysis, dict)
        self.assertIsNone(analysis["suggestion"])
        self.assertIsNone(projection["golden_handoff"])
        self.assertIs(analysis["decision_eligible"], False)
        self.assertIs(analysis["auto_apply"], False)
        self.assertEqual(analysis["decision_effect"], "none")
        for name in (
            "decision_eligible",
            "auto_apply",
            "golden_validation_performed",
            "golden_protocol_eligible",
            "changes_applied",
        ):
            self.assertIs(projection[name], False)

    async def test_missing_required_family_is_audited_as_insufficient(self) -> None:
        bodies = tuple(
            b"\n".join(
                line
                for line in body.splitlines()
                if b"vllm:request_queue_time_seconds" not in line
            )
            + b"\n"
            for body in _fixture_bodies()
        )
        outcome, _ = await self._run_fixture(bodies)
        projection = outcome.supplementary  # type: ignore[attr-defined]
        self.assertEqual(outcome.analysis_status, "insufficient_evidence")  # type: ignore[attr-defined]
        self.assert_report_analysis_reconciliation(
            outcome,
            expected_concurrency=16,
        )
        self.assertIsNone(projection["golden_handoff"])
        analysis = projection["analysis"]
        assert isinstance(analysis, dict)
        self.assertIn("missing_required_metric", analysis["quality_reasons"])
        self.assertIsNone(analysis["suggestion"])
        for name in (
            "decision_eligible",
            "auto_apply",
            "golden_validation_performed",
            "golden_protocol_eligible",
            "changes_applied",
        ):
            self.assertIs(projection[name], False)

    async def test_absent_same_deployment_exclusivity_attestation_is_insufficient(
        self,
    ) -> None:
        outcome, _ = await self._run_fixture(traffic_scope="unconfirmed")
        self.assertEqual(outcome.analysis_status, "insufficient_evidence")  # type: ignore[attr-defined]
        self.assert_report_analysis_reconciliation(
            outcome,
            expected_concurrency=16,
        )
        projection = outcome.supplementary  # type: ignore[attr-defined]
        self.assertIsNone(projection["golden_handoff"])
        analysis = projection["analysis"]
        assert isinstance(analysis, dict)
        self.assertIn(
            "traffic_scope_not_exclusive",
            analysis["quality_reasons"],
        )
        self.assertIsNone(analysis["suggestion"])
        for name in (
            "decision_eligible",
            "auto_apply",
            "golden_validation_performed",
            "golden_protocol_eligible",
            "changes_applied",
        ):
            self.assertIs(projection[name], False)

    async def test_valid_smoke_slo_miss_remains_suggestion_only(self) -> None:
        slo_config = replace(_config(), p95_slo_ms=0.000001)
        outcome, _ = await self._run_fixture(config=slo_config)

        report = outcome.report  # type: ignore[attr-defined]
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["conditions"][0]["valid"])
        self.assertEqual(
            report["best_tested"],
            {
                "field": "best_tested_concurrency",
                "available": False,
                "state": "inconclusive",
                "reason": "no_valid_condition_meets_slo",
                "optimum_found": False,
            },
        )
        self.assertIs(report["decision_eligible"], False)

        projection = outcome.supplementary  # type: ignore[attr-defined]
        self.assertEqual(outcome.analysis_status, "suggestion_available")  # type: ignore[attr-defined]
        self.assert_report_analysis_reconciliation(
            outcome,
            expected_concurrency=16,
        )
        self.assert_hard_false_locks(projection)

    async def test_unrelated_completion_contamination_suppresses_suggestion(
        self,
    ) -> None:
        outcome, fixture = await self._run_connected_fixture(
            _ConnectedVllmFixture(contaminate_final=True)
        )
        self.assertEqual(fixture.finished_requests, 40)
        self.assertEqual(outcome.analysis_status, "insufficient_evidence")  # type: ignore[attr-defined]
        self.assert_report_analysis_reconciliation(
            outcome,
            expected_concurrency=16,
            expected_exporter_finished=41,
        )
        projection = outcome.supplementary  # type: ignore[attr-defined]
        self.assertIsNone(projection["golden_handoff"])
        analysis = projection["analysis"]
        assert isinstance(analysis, dict)
        self.assertIn(
            "exporter_request_count_mismatch",
            analysis["quality_reasons"],
        )
        self.assertIsNone(analysis["suggestion"])
        for name in (
            "decision_eligible",
            "auto_apply",
            "golden_validation_performed",
            "golden_protocol_eligible",
            "changes_applied",
        ):
            self.assertIs(projection[name], False)

    async def test_exporter_counter_reset_fails_before_any_projection(self) -> None:
        with self.assertRaises(ExperimentalTuningError) as raised:
            await self._run_connected_fixture(
                _ConnectedVllmFixture(reset_final=True)
            )
        _assert_fixed_error(
            self,
            "experimental_metrics_window_invalid",
            raised.exception,
        )


class ExperimentalTuningFailureTests(unittest.IsolatedAsyncioTestCase):
    async def _expect_failure(
        self,
        expected: str,
        collector: object,
        controlled: _ControlledRun,
        *,
        clock_values: tuple[float, ...] = (0.0, 2.0),
    ) -> None:
        clock = iter(clock_values)
        with patch.object(
            experimental_module.asyncio,
            "sleep",
            side_effect=controlled.sampling_sleep,
        ):
            with self.assertRaises(ExperimentalTuningError) as raised:
                await run_experimental_tuning(
                    _config(),
                    _MEASURED_PROMPTS,
                    _WARMUP_PROMPTS,
                    collector=collector,  # type: ignore[arg-type]
                    traffic_scope="operator_attested_exclusive",
                    progress=RunProgress(),
                    run_traffic=controlled.traffic,
                    sample_interval_seconds=1.0,
                    monotonic=lambda: next(clock),
                )
        _assert_fixed_error(self, expected, raised.exception)

    async def test_pre_scrape_failure_starts_no_traffic(self) -> None:
        controlled = _ControlledRun()
        collector = _SequenceCollector([ValueError(_PRIVATE_STAGE_PAYLOAD)])
        await self._expect_failure(
            "experimental_metrics_collection_failed",
            collector,
            controlled,
            clock_values=(0.0,),
        )
        self.assertEqual(collector.calls, 1)
        self.assertFalse(controlled.started.is_set())
        self.assertFalse(controlled.cancelled.is_set())

    async def test_malformed_initial_snapshot_fails_before_traffic(self) -> None:
        from throttle.server_metrics import MetricsSnapshot

        controlled = _ControlledRun()
        collector = _SequenceCollector([MetricsSnapshot(-1, (), ())])
        await self._expect_failure(
            "experimental_metrics_snapshot_invalid",
            collector,
            controlled,
            clock_values=(0.0,),
        )
        self.assertEqual(collector.calls, 1)
        self.assertFalse(controlled.started.is_set())
        self.assertFalse(controlled.cancelled.is_set())

    async def test_mid_scrape_failure_cancels_active_traffic(self) -> None:
        snapshots = []
        from throttle.server_metrics import parse_vllm_metrics

        snapshots.append(parse_vllm_metrics(_fixture_bodies()[0]))
        collector = _SequenceCollector(
            [snapshots[0], ValueError(_PRIVATE_STAGE_PAYLOAD)]
        )
        controlled = _ControlledRun()
        await self._expect_failure(
            "experimental_metrics_collection_failed",
            collector,
            controlled,
            clock_values=(0.0, 2.0),
        )
        self.assertTrue(controlled.started.is_set())
        self.assertTrue(controlled.cancelled.is_set())
        self.assertTrue(controlled.finished.is_set())

    async def test_missing_snapshot_cancels_active_traffic(self) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        before = parse_vllm_metrics(_fixture_bodies()[0])
        collector = _SequenceCollector([before, None])
        controlled = _ControlledRun()
        await self._expect_failure(
            "experimental_metrics_snapshot_invalid",
            collector,
            controlled,
        )
        self.assertTrue(controlled.cancelled.is_set())

    async def test_excessive_sample_gap_cancels_active_traffic(self) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        before = parse_vllm_metrics(_fixture_bodies()[0])
        collector = _SequenceCollector([before])
        controlled = _ControlledRun()
        await self._expect_failure(
            "experimental_metrics_sample_gap_exceeded",
            collector,
            controlled,
            clock_values=(0.0, 5.0000001),
        )
        self.assertTrue(controlled.cancelled.is_set())

    async def test_cumulative_scrape_budget_fails_before_window_derivation(
        self,
    ) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        exposition = "\n".join(
            "vllm:num_requests_running"
            f'{{engine="0",model_name="fixture-{index}"}} 0'
            for index in range(4_000)
        )
        snapshot = parse_vllm_metrics((exposition + "\n").encode("utf-8"))
        self.assertEqual(snapshot.recognized_samples, 4_000)
        collector = _SequenceCollector([snapshot] * 17)
        controlled = _ControlledRun(release_after=100)
        clock = iter(float(index) for index in range(17))
        with (
            patch.object(
                experimental_module.asyncio,
                "sleep",
                side_effect=controlled.sampling_sleep,
            ),
            patch.object(
                experimental_module,
                "derive_metrics_window",
                side_effect=AssertionError(
                    "oversized scrape window reached derivation"
                ),
            ) as derive,
        ):
            with self.assertRaises(ExperimentalTuningError) as raised:
                await run_experimental_tuning(
                    _config(),
                    _MEASURED_PROMPTS,
                    _WARMUP_PROMPTS,
                    collector=collector,  # type: ignore[arg-type]
                    traffic_scope="operator_attested_exclusive",
                    progress=RunProgress(),
                    run_traffic=controlled.traffic,
                    monotonic=lambda: next(clock),
                )
        _assert_fixed_error(
            self,
            "experimental_metrics_window_too_large",
            raised.exception,
        )
        self.assertEqual(collector.calls, 17)
        self.assertTrue(controlled.started.is_set())
        self.assertTrue(controlled.cancelled.is_set())
        self.assertTrue(controlled.finished.is_set())
        derive.assert_not_called()

    async def test_outer_cancellation_cancels_active_traffic(self) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        before = parse_vllm_metrics(_fixture_bodies()[0])
        collector = _SequenceCollector([before])
        controlled = _ControlledRun()
        sampling_started = asyncio.Event()

        async def blocked_sleep(_: float) -> None:
            sampling_started.set()
            await asyncio.Event().wait()

        with patch.object(
            experimental_module.asyncio,
            "sleep",
            side_effect=blocked_sleep,
        ):
            task = asyncio.create_task(
                run_experimental_tuning(
                    _config(),
                    _MEASURED_PROMPTS,
                    _WARMUP_PROMPTS,
                    collector=collector,  # type: ignore[arg-type]
                    traffic_scope="operator_attested_exclusive",
                    progress=RunProgress(),
                    run_traffic=controlled.traffic,
                )
            )
            while not sampling_started.is_set():
                await _REAL_ASYNCIO_SLEEP(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(controlled.cancelled.is_set())
        self.assertTrue(controlled.finished.is_set())

    async def test_invalid_run_report_never_reaches_window_analysis_or_audit(
        self,
    ) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        def incomplete(report: dict[str, Any]) -> None:
            report["status"] = "stopped"

        def invalid_condition(report: dict[str, Any]) -> None:
            report["conditions"][0]["valid"] = False

        def mismatched_manifest(report: dict[str, Any]) -> None:
            report["manifest"]["engine"]["effective_flags"][
                "max_num_seqs"
            ] = "9"

        def mismatched_workload(report: dict[str, Any]) -> None:
            report["manifest"]["workload"]["measured_sha256"] = "0" * 64

        def mismatched_request(report: dict[str, Any]) -> None:
            report["manifest"]["request"]["max_tokens"] = 129

        def mismatched_traffic(report: dict[str, Any]) -> None:
            report["manifest"]["traffic"]["requests_per_block"] = 36

        def mismatched_block(report: dict[str, Any]) -> None:
            report["conditions"][0]["blocks"][0]["valid"] = False

        for name, mutate in (
            ("incomplete", incomplete),
            ("invalid_condition", invalid_condition),
            ("mismatched_manifest", mismatched_manifest),
            ("mismatched_workload", mismatched_workload),
            ("mismatched_request", mismatched_request),
            ("mismatched_traffic", mismatched_traffic),
            ("mismatched_block", mismatched_block),
        ):
            with self.subTest(case=name):
                report = await _full_smoke_report()
                mutate(report)
                snapshots = [
                    parse_vllm_metrics(body) for body in _fixture_bodies()[:4]
                ]
                collector = _SequenceCollector(snapshots)
                controlled = _ControlledRun(report)
                clock = iter((0.0, 2.0, 4.0, 6.0))
                with (
                    patch.object(
                        experimental_module.asyncio,
                        "sleep",
                        side_effect=controlled.sampling_sleep,
                    ),
                    patch.object(
                        experimental_module,
                        "derive_metrics_window",
                        side_effect=AssertionError(
                            "invalid report reached metrics derivation"
                        ),
                    ) as derive,
                    patch.object(
                        experimental_module,
                        "analyze_bottleneck",
                        side_effect=AssertionError(
                            "invalid report reached analyzer"
                        ),
                    ) as analyze,
                    patch.object(
                        experimental_module,
                        "audit_agent_outputs",
                        side_effect=AssertionError(
                            "invalid report reached safety audit"
                        ),
                    ) as audit,
                ):
                    with self.assertRaises(ExperimentalTuningError) as raised:
                        await run_experimental_tuning(
                            _config(),
                            _MEASURED_PROMPTS,
                            _WARMUP_PROMPTS,
                            collector=collector,  # type: ignore[arg-type]
                            traffic_scope="operator_attested_exclusive",
                            progress=RunProgress(),
                            run_traffic=controlled.traffic,
                            monotonic=lambda: next(clock),
                        )
                _assert_fixed_error(
                    self,
                    "experimental_run_report_invalid",
                    raised.exception,
                )
                derive.assert_not_called()
                analyze.assert_not_called()
                audit.assert_not_called()

    async def test_traffic_runner_cannot_redefine_workload_by_mutation(
        self,
    ) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        mutated_prompts = (
            ({"role": "user", "content": "runner-mutated prompt"},),
        )
        mutated_report = await _full_smoke_report(
            prompts=mutated_prompts,
            warmup_prompts=_WARMUP_PROMPTS,
        )
        collector = _SequenceCollector(
            [parse_vllm_metrics(_fixture_bodies()[0])]
        )
        runner_received_copy = False

        async def malicious_runner(
            config: object,
            prompts: object,
            warmup_prompts: object,
            *,
            progress: object,
        ) -> dict[str, Any]:
            nonlocal runner_received_copy
            del config, warmup_prompts, progress
            mutable_prompts: Any = prompts
            mutable_prompts[0][0]["content"] = "runner-mutated prompt"
            runner_received_copy = True
            return mutated_report

        async def yield_to_runner(_: float) -> None:
            await _REAL_ASYNCIO_SLEEP(0)

        with (
            patch.object(
                experimental_module.asyncio,
                "sleep",
                side_effect=yield_to_runner,
            ),
            patch.object(
                experimental_module,
                "derive_metrics_window",
                side_effect=AssertionError(
                    "mutated workload report reached metrics derivation"
                ),
            ) as derive,
            patch.object(
                experimental_module,
                "analyze_bottleneck",
                side_effect=AssertionError(
                    "mutated workload report reached analyzer"
                ),
            ) as analyze,
            patch.object(
                experimental_module,
                "audit_agent_outputs",
                side_effect=AssertionError(
                    "mutated workload report reached safety audit"
                ),
            ) as audit,
        ):
            with self.assertRaises(ExperimentalTuningError) as raised:
                await run_experimental_tuning(
                    _config(),
                    _MEASURED_PROMPTS,
                    _WARMUP_PROMPTS,
                    collector=collector,  # type: ignore[arg-type]
                    traffic_scope="operator_attested_exclusive",
                    progress=RunProgress(),
                    run_traffic=malicious_runner,  # type: ignore[arg-type]
                    monotonic=lambda: 0.0,
                )
        _assert_fixed_error(
            self,
            "experimental_run_report_invalid",
            raised.exception,
        )
        self.assertTrue(runner_received_copy)
        self.assertEqual(
            _MEASURED_PROMPTS[0][0]["content"],
            "public measured prompt",
        )
        self.assertEqual(collector.calls, 1)
        derive.assert_not_called()
        analyze.assert_not_called()
        audit.assert_not_called()

    async def test_malformed_derived_window_never_reaches_a_projection(self) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        snapshots = [parse_vllm_metrics(body) for body in _fixture_bodies()]
        collector = _SequenceCollector(snapshots)
        controlled = _ControlledRun(await _full_smoke_report())
        clock = iter((0.0, 2.0, 4.0, 6.0, 8.0))
        with (
            patch.object(
                experimental_module.asyncio,
                "sleep",
                side_effect=controlled.sampling_sleep,
            ),
            patch.object(
                experimental_module,
                "derive_metrics_window",
                return_value={"private": _PRIVATE_STAGE_PAYLOAD},
            ),
        ):
            with self.assertRaises(ExperimentalTuningError) as raised:
                await run_experimental_tuning(
                    _config(),
                    _MEASURED_PROMPTS,
                    _WARMUP_PROMPTS,
                    collector=collector,  # type: ignore[arg-type]
                    traffic_scope="operator_attested_exclusive",
                    progress=RunProgress(),
                    run_traffic=controlled.traffic,
                    monotonic=lambda: next(clock),
                )
        _assert_fixed_error(
            self, "experimental_analysis_failed", raised.exception
        )

    async def test_missing_or_tampered_analysis_is_rejected_by_safety(self) -> None:
        from throttle.bottleneck_analysis import analyze_bottleneck as real_analyze
        from throttle.server_metrics import parse_vllm_metrics

        cases = (
            (None, "experimental_safety_validation_failed"),
            ({"private": _PRIVATE_STAGE_PAYLOAD}, "experimental_safety_validation_failed"),
        )
        for supplied, expected in cases:
            snapshots = [parse_vllm_metrics(body) for body in _fixture_bodies()]
            collector = _SequenceCollector(snapshots)
            controlled = _ControlledRun(await _full_smoke_report())
            clock = iter((0.0, 2.0, 4.0, 6.0, 8.0))
            with self.subTest(supplied=type(supplied).__name__):
                with (
                    patch.object(
                        experimental_module.asyncio,
                        "sleep",
                        side_effect=controlled.sampling_sleep,
                    ),
                    patch.object(
                        experimental_module,
                        "analyze_bottleneck",
                        return_value=supplied,
                    ),
                ):
                    with self.assertRaises(ExperimentalTuningError) as raised:
                        await run_experimental_tuning(
                            _config(),
                            _MEASURED_PROMPTS,
                            _WARMUP_PROMPTS,
                            collector=collector,  # type: ignore[arg-type]
                            traffic_scope="operator_attested_exclusive",
                            progress=RunProgress(),
                            run_traffic=controlled.traffic,
                            monotonic=lambda: next(clock),
                        )
                _assert_fixed_error(self, expected, raised.exception)
        self.assertIsNotNone(real_analyze)

    async def test_missing_or_tampered_safety_result_is_never_rendered(self) -> None:
        from throttle.server_metrics import parse_vllm_metrics

        for supplied in (None, {"private": _PRIVATE_STAGE_PAYLOAD}):
            snapshots = [parse_vllm_metrics(body) for body in _fixture_bodies()]
            collector = _SequenceCollector(snapshots)
            controlled = _ControlledRun(await _full_smoke_report())
            clock = iter((0.0, 2.0, 4.0, 6.0, 8.0))
            with self.subTest(supplied=type(supplied).__name__):
                with (
                    patch.object(
                        experimental_module.asyncio,
                        "sleep",
                        side_effect=controlled.sampling_sleep,
                    ),
                    patch.object(
                        experimental_module,
                        "audit_agent_outputs",
                        return_value=supplied,
                    ),
                ):
                    with self.assertRaises(ExperimentalTuningError) as raised:
                        await run_experimental_tuning(
                            _config(),
                            _MEASURED_PROMPTS,
                            _WARMUP_PROMPTS,
                            collector=collector,  # type: ignore[arg-type]
                            traffic_scope="operator_attested_exclusive",
                            progress=RunProgress(),
                            run_traffic=controlled.traffic,
                            monotonic=lambda: next(clock),
                        )
                _assert_fixed_error(
                    self,
                    "experimental_safety_projection_failed",
                    raised.exception,
                )


if __name__ == "__main__":
    unittest.main()
