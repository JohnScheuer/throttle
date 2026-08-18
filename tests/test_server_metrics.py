from __future__ import annotations

import asyncio
import json
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from throttle.server_metrics import (
    MAX_LABELS_PER_SAMPLE,
    MAX_LINE_BYTES,
    MAX_LINES,
    MAX_OBSERVATIONS,
    MAX_RECOGNIZED_SAMPLES,
    MAX_RESPONSE_BYTES,
    MAX_WINDOW_RECOGNIZED_SAMPLES,
    MetricsCollectionError,
    PrometheusMetricsCollector,
    derive_metrics_window,
    parse_vllm_metrics,
)


class _DelayedStream(httpx.AsyncByteStream):
    async def __aiter__(self):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.05)
        yield b"vllm:generation_tokens_total 1\n"


class _ChunkedOversizeStream(httpx.AsyncByteStream):
    async def __aiter__(self):  # type: ignore[no-untyped-def]
        yield b"x" * MAX_RESPONSE_BYTES
        yield b"x"


class _EqualSecret(str):
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


def _line(name: str, value: int | float, **labels: str) -> str:
    if labels:
        rendered = ",".join(
            f'{key}="{item}"' for key, item in labels.items()
        )
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


def _snapshot(
    *,
    request: tuple[int, int] = (10, 5),
    generated: tuple[int, int] = (1_000, 500),
    prompt: tuple[int, int] = (300, 200),
    preemptions: tuple[int, int] = (1, 0),
    histogram_values: dict[str, tuple[tuple[float, int], tuple[float, int]]]
    | None = None,
    running: tuple[int, int] = (1, 1),
    waiting: tuple[int, int] = (0, 0),
    swapped: tuple[int, int] = (0, 0),
    kv: tuple[float, float] = (0.2, 0.3),
    secret_label: str = "private-model-name",
) -> object:
    histogram_values = histogram_values or {
        "vllm:time_to_first_token_seconds": ((2.0, 10), (1.0, 5)),
        "vllm:request_time_per_output_token_seconds": (
            (0.5, 10),
            (0.25, 5),
        ),
        "vllm:e2e_request_latency_seconds": ((10.0, 10), (5.0, 5)),
        "vllm:request_queue_time_seconds": ((1.0, 10), (0.5, 5)),
        "vllm:request_prefill_time_seconds": ((2.0, 10), (1.0, 5)),
        "vllm:request_decode_time_seconds": ((8.0, 10), (4.0, 5)),
    }
    lines = ["# HELP ignored private help text", "# TYPE ignored gauge"]
    for index, engine in enumerate(("worker-a", "worker-b")):
        common = {"engine": engine, "model": secret_label}
        lines.extend(
            (
                _line(
                    "vllm:request_success_total",
                    request[index],
                    **common,
                    finished_reason="stop",
                ),
                _line(
                    "vllm:generation_tokens_total",
                    generated[index],
                    **common,
                ),
                _line(
                    "vllm:prompt_tokens_total", prompt[index], **common
                ),
                _line(
                    "vllm:num_preemptions_total",
                    preemptions[index],
                    **common,
                ),
                _line(
                    "vllm:num_requests_running", running[index], **common
                ),
                _line(
                    "vllm:num_requests_waiting", waiting[index], **common
                ),
                _line(
                    "vllm:num_requests_swapped", swapped[index], **common
                ),
                _line("vllm:kv_cache_usage_perc", kv[index], **common),
            )
        )
        for base, values in histogram_values.items():
            total, count = values[index]
            lines.append(_line(f"{base}_sum", total, **common))
            lines.append(_line(f"{base}_count", count, **common))
    lines.append('unknown_metric{credential="do-not-persist"} 7')
    return parse_vllm_metrics("\n".join(lines) + "\n")


class ParserAndDerivationTests(unittest.TestCase):
    def assert_code(self, expected: str, callable_: object) -> None:
        with self.assertRaises(MetricsCollectionError) as raised:
            callable_()  # type: ignore[operator]
        self.assertEqual(str(raised.exception), expected)
        self.assertEqual(raised.exception.code, expected)

    def test_labeled_multi_worker_window_is_aggregated_after_deltas(self) -> None:
        before = _snapshot()
        observation = _snapshot(
            request=(12, 6),
            generated=(1_040, 530),
            prompt=(320, 215),
            preemptions=(2, 0),
            running=(4, 3),
            waiting=(1, 2),
            swapped=(0, 1),
            kv=(0.7, 0.8),
        )
        after = _snapshot(
            request=(14, 7),
            generated=(1_120, 580),
            prompt=(360, 240),
            preemptions=(3, 0),
            running=(0, 0),
            waiting=(0, 0),
            swapped=(0, 0),
            kv=(0.1, 0.2),
            histogram_values={
                "vllm:time_to_first_token_seconds": (
                    (2.8, 14),
                    (1.4, 7),
                ),
                "vllm:request_time_per_output_token_seconds": (
                    (0.7, 14),
                    (0.35, 7),
                ),
                "vllm:e2e_request_latency_seconds": (
                    (14.0, 14),
                    (7.0, 7),
                ),
                "vllm:request_queue_time_seconds": (
                    (1.4, 14),
                    (0.7, 7),
                ),
                "vllm:request_prefill_time_seconds": (
                    (2.8, 14),
                    (1.4, 7),
                ),
                "vllm:request_decode_time_seconds": (
                    (11.2, 14),
                    (5.6, 7),
                ),
            },
        )

        window = derive_metrics_window(
            before, after, 2.0, observations=(observation,)
        )
        self.assertEqual(window.requests_finished, 6)
        self.assertEqual(window.output_tokens, 200)
        self.assertEqual(window.prompt_tokens, 100)
        self.assertEqual(window.preemptions, 2)
        self.assertEqual(window.finished_requests_per_second, 3.0)
        self.assertEqual(window.output_tokens_per_second, 100.0)
        self.assertEqual(window.prompt_tokens_per_second, 50.0)
        self.assertAlmostEqual(window.mean_ttft_ms or 0, 200.0)
        self.assertAlmostEqual(window.mean_tpot_ms or 0, 50.0)
        self.assertIsNone(window.mean_inter_token_latency_ms)
        self.assertAlmostEqual(window.mean_e2e_ms or 0, 1_000.0)
        self.assertAlmostEqual(window.mean_queue_time_ms or 0, 100.0)
        self.assertAlmostEqual(window.mean_prefill_time_ms or 0, 200.0)
        self.assertAlmostEqual(window.mean_decode_time_ms or 0, 800.0)
        self.assertEqual(window.max_requests_running, 7)
        self.assertEqual(window.max_requests_waiting, 3)
        self.assertEqual(window.max_requests_swapped, 1)
        self.assertEqual(window.max_kv_cache_usage_fraction, 0.8)
        self.assertEqual(window.observations, 3)
        self.assertEqual(window.decision_effect, "none")
        rendered = json.dumps(window.to_public_dict(), allow_nan=False)
        self.assertNotIn("private-model-name", rendered)
        self.assertNotIn("worker-a", rendered)
        self.assertNotIn("do-not-persist", rendered)
        self.assertNotIn("private-model-name", repr(before))

    def test_finish_reason_series_are_summed(self) -> None:
        before = parse_vllm_metrics(
            '\n'.join(
                (
                    'vllm:request_success_total{finished_reason="stop"} 10',
                    'vllm:request_success_total{finished_reason="length"} 3',
                )
            )
        )
        after = parse_vllm_metrics(
            '\n'.join(
                (
                    'vllm:request_success_total{finished_reason="stop"} 14',
                    'vllm:request_success_total{finished_reason="length"} 5',
                )
            )
        )
        result = derive_metrics_window(before, after, 2)
        self.assertEqual(result.requests_finished, 6)
        self.assertEqual(result.finished_requests_per_second, 3)
        public = result.to_public_dict()
        self.assertEqual(public["finished_requests_per_second"], 3)
        self.assertNotIn("requests_per_second", public)

    def test_window_projection_rejects_forged_values_and_text(self) -> None:
        before = parse_vllm_metrics("vllm:generation_tokens_total 1")
        after = parse_vllm_metrics("vllm:generation_tokens_total 2")
        window = derive_metrics_window(before, after, 1)
        private = "https://private.example/api_key=secret"
        mutations = (
            {"elapsed_seconds": math.nan},
            {"scope": private},
            {"source_families": (private,)},
            {"caveats": (private,)},
        )
        for changes in mutations:
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(MetricsCollectionError) as raised:
                    replace(window, **changes)
                self.assertEqual(
                    raised.exception.code, "metrics_invalid_window"
                )
                self.assertNotIn(private, str(raised.exception))
        tampered = derive_metrics_window(before, after, 1)
        object.__setattr__(tampered, "scope", private)
        with self.assertRaises(MetricsCollectionError) as raised:
            tampered.to_public_dict()
        self.assertEqual(raised.exception.code, "metrics_invalid_window")
        self.assertNotIn(private, str(raised.exception))
        malformed = derive_metrics_window(before, after, 1)
        object.__setattr__(malformed, "source_families", ([],))
        with self.assertRaises(MetricsCollectionError) as raised:
            malformed.to_public_dict()
        self.assertEqual(raised.exception.code, "metrics_invalid_window")

        equal_secret = _EqualSecret(private)
        forged_caveats = (equal_secret,) * len(window.caveats)
        with self.assertRaises(MetricsCollectionError) as raised:
            replace(window, caveats=forged_caveats)
        self.assertEqual(raised.exception.code, "metrics_invalid_window")
        self.assertNotIn(private, str(raised.exception))

    def test_missing_is_distinct_from_a_real_zero(self) -> None:
        body = "\n".join(
            (
                "vllm:request_success_total 0",
                "vllm:num_requests_running 0",
                "vllm:gpu_cache_usage_perc 0",
                "vllm:time_per_output_token_seconds_sum 0",
                "vllm:time_per_output_token_seconds_count 0",
            )
        )
        before = parse_vllm_metrics(body)
        after = parse_vllm_metrics(body)
        result = derive_metrics_window(before, after, 1)
        self.assertEqual(result.requests_finished, 0)
        self.assertEqual(result.finished_requests_per_second, 0)
        self.assertEqual(result.max_requests_running, 0)
        self.assertEqual(result.max_kv_cache_usage_fraction, 0)
        self.assertIsNone(result.output_tokens)
        self.assertIsNone(result.output_tokens_per_second)
        self.assertIsNone(result.mean_tpot_ms)
        self.assertIsNone(result.mean_inter_token_latency_ms)
        self.assertIn("vllm:gpu_cache_usage_perc", result.source_families)

    def test_distinct_tpot_and_itl_and_current_alias_precedence(self) -> None:
        before = parse_vllm_metrics(
            "\n".join(
                (
                    "vllm:num_preemptions_total 10",
                    "vllm:num_preemptions 1000",
                    "vllm:kv_cache_usage_perc 0.2",
                    "vllm:gpu_cache_usage_perc 0.9",
                    "vllm:request_time_per_output_token_seconds_sum 1",
                    "vllm:request_time_per_output_token_seconds_count 10",
                    "vllm:inter_token_latency_seconds_sum 2",
                    "vllm:inter_token_latency_seconds_count 10",
                    "vllm:time_per_output_token_seconds_sum 50",
                    "vllm:time_per_output_token_seconds_count 10",
                )
            )
        )
        after = parse_vllm_metrics(
            "\n".join(
                (
                    "vllm:num_preemptions_total 12",
                    "vllm:num_preemptions 2000",
                    "vllm:kv_cache_usage_perc 0.3",
                    "vllm:gpu_cache_usage_perc 1",
                    "vllm:request_time_per_output_token_seconds_sum 1.2",
                    "vllm:request_time_per_output_token_seconds_count 12",
                    "vllm:inter_token_latency_seconds_sum 2.1",
                    "vllm:inter_token_latency_seconds_count 12",
                    "vllm:time_per_output_token_seconds_sum 100",
                    "vllm:time_per_output_token_seconds_count 12",
                )
            )
        )
        result = derive_metrics_window(before, after, 1)
        self.assertEqual(result.preemptions, 2)
        self.assertAlmostEqual(result.mean_tpot_ms or 0, 100)
        self.assertAlmostEqual(
            result.mean_inter_token_latency_ms or 0, 50
        )
        self.assertAlmostEqual(result.max_kv_cache_usage_fraction or 0, 0.3)
        self.assertNotIn("vllm:num_preemptions", result.source_families)
        self.assertNotIn("vllm:gpu_cache_usage_perc", result.source_families)
        self.assertNotIn(
            "vllm:time_per_output_token_seconds_sum",
            result.source_families,
        )

    def test_alias_change_is_rejected(self) -> None:
        before = parse_vllm_metrics("vllm:kv_cache_usage_perc 0.2")
        after = parse_vllm_metrics("vllm:gpu_cache_usage_perc 0.3")
        self.assert_code(
            "metrics_alias_or_availability_changed",
            lambda: derive_metrics_window(before, after, 1),
        )

    def test_per_series_reset_cannot_be_hidden_by_aggregate_growth(self) -> None:
        before = parse_vllm_metrics(
            "\n".join(
                (
                    'vllm:generation_tokens_total{engine="a"} 100',
                    'vllm:generation_tokens_total{engine="b"} 100',
                )
            )
        )
        after = parse_vllm_metrics(
            "\n".join(
                (
                    'vllm:generation_tokens_total{engine="a"} 250',
                    'vllm:generation_tokens_total{engine="b"} 1',
                )
            )
        )
        self.assert_code(
            "metrics_counter_reset",
            lambda: derive_metrics_window(before, after, 1),
        )

    def test_intermediate_reset_or_topology_change_cannot_be_hidden(self) -> None:
        before = parse_vllm_metrics(
            'vllm:generation_tokens_total{engine="a"} 100'
        )
        reset = parse_vllm_metrics(
            'vllm:generation_tokens_total{engine="a"} 1'
        )
        recovered = parse_vllm_metrics(
            'vllm:generation_tokens_total{engine="a"} 120'
        )
        self.assert_code(
            "metrics_counter_reset",
            lambda: derive_metrics_window(
                before, recovered, 1, observations=(reset,)
            ),
        )
        changed = parse_vllm_metrics(
            'vllm:generation_tokens_total{engine="b"} 110'
        )
        self.assert_code(
            "metrics_series_topology_changed",
            lambda: derive_metrics_window(
                before, recovered, 1, observations=(changed,)
            ),
        )

    def test_series_identity_encoding_is_unambiguous(self) -> None:
        before = parse_vllm_metrics(
            'vllm:generation_tokens_total{a="x\x1fb\x1ey"} 1'
        )
        after = parse_vllm_metrics(
            'vllm:generation_tokens_total{a="x",b="y"} 2'
        )
        self.assert_code(
            "metrics_series_topology_changed",
            lambda: derive_metrics_window(before, after, 1),
        )

    def test_series_topology_changes_are_rejected(self) -> None:
        before = parse_vllm_metrics(
            'vllm:generation_tokens_total{engine="a"} 10'
        )
        after = parse_vllm_metrics(
            'vllm:generation_tokens_total{engine="b"} 11'
        )
        self.assert_code(
            "metrics_series_topology_changed",
            lambda: derive_metrics_window(before, after, 1),
        )

    def test_quoted_label_grammar_and_unknown_metadata_are_discarded(self) -> None:
        body = (
            'vllm:generation_tokens_total{model="private model, '
            'quoted \\" value",path="line\\nnext\\\\tail"} 7\n'
            'unknown_family{api_key="private-secret"} 99 123\n'
            "# EOF\n"
        )
        snapshot = parse_vllm_metrics(body)
        self.assertEqual(snapshot.recognized_samples, 1)
        rendered = repr(snapshot) + json.dumps(snapshot.public_summary())
        for private in ("private model", "private-secret", "line", "tail"):
            self.assertNotIn(private, rendered)

    def test_timestamp_is_accepted_and_comments_are_ignored(self) -> None:
        snapshot = parse_vllm_metrics(
            "# HELP private text\n"
            'vllm:generation_tokens_total{engine="a"} 7 123456\n'
            "# EOF\n"
        )
        self.assertEqual(snapshot.recognized_samples, 1)

    def test_unknown_families_are_discarded_without_parsing_payloads(self) -> None:
        private = "private-trace-value"
        snapshot = parse_vllm_metrics(
            "unknown_nan NaN\n"
            f'unknown_exemplar 1 # {{trace_id="{private}"}} 2\n'
            "vllm:generation_tokens_total 7\n"
        )
        self.assertEqual(snapshot.recognized_samples, 1)
        self.assertNotIn(private, repr(snapshot))

    def test_malformed_and_unsafe_values_fail_with_fixed_codes(self) -> None:
        cases = {
            "metrics_invalid_utf8": b"\xff",
            "metrics_invalid_number": "vllm:generation_tokens_total nope",
            "metrics_nonfinite_value": "vllm:generation_tokens_total NaN",
            "metrics_negative_value": "vllm:generation_tokens_total -1",
            "metrics_nonintegral_count": (
                "vllm:generation_tokens_total 1.5"
            ),
            "metrics_invalid_kv_fraction": "vllm:kv_cache_usage_perc 1.1",
            "metrics_precise_invalid_kv_fraction": (
                "vllm:kv_cache_usage_perc 1.0000000000000000001"
            ),
            "metrics_precise_nonintegral_count": (
                "vllm:generation_tokens_total 1.0000000000000000001"
            ),
            "metrics_value_out_of_range": (
                "vllm:time_to_first_token_seconds_sum 9007199254740992"
            ),
            "metrics_value_underflow": (
                "vllm:time_to_first_token_seconds_sum 1e-9999"
            ),
            "metrics_duplicate_label": (
                'vllm:generation_tokens_total{a="1",a="2"} 1'
            ),
            "metrics_duplicate_series": (
                "vllm:generation_tokens_total 1\n"
                "vllm:generation_tokens_total 2"
            ),
            "metrics_malformed_sample_escape": (
                'vllm:generation_tokens_total{a="bad\\tvalue"} 1'
            ),
        }
        private = "private-payload-value"
        for expected, value in cases.items():
            with self.subTest(expected=expected):
                actual = (
                    "metrics_malformed_sample"
                    if expected == "metrics_malformed_sample_escape"
                    else (
                        "metrics_invalid_kv_fraction"
                        if expected == "metrics_precise_invalid_kv_fraction"
                        else (
                            "metrics_nonintegral_count"
                            if expected
                            == "metrics_precise_nonintegral_count"
                            else expected
                        )
                    )
                )
                with self.assertRaises(MetricsCollectionError) as raised:
                    parse_vllm_metrics(value)
                self.assertEqual(str(raised.exception), actual)
                self.assertNotIn(private, str(raised.exception))

    def test_parser_resource_caps(self) -> None:
        cases = (
            (
                "metrics_response_too_large",
                b"x" * (MAX_RESPONSE_BYTES + 1),
            ),
            (
                "metrics_line_too_long",
                "x" * (MAX_LINE_BYTES + 1),
            ),
            (
                "metrics_too_many_lines",
                "\n" * (MAX_LINES + 1),
            ),
            (
                "metrics_too_many_labels",
                "vllm:generation_tokens_total{"
                + ",".join(
                    f'l{index}="x"'
                    for index in range(MAX_LABELS_PER_SAMPLE + 1)
                )
                + "} 1",
            ),
            (
                "metrics_too_many_recognized_samples",
                "\n".join(
                    _line(
                        "vllm:generation_tokens_total", 1, series=str(index)
                    )
                    for index in range(MAX_RECOGNIZED_SAMPLES + 1)
                ),
            ),
        )
        for expected, body in cases:
            with self.subTest(expected=expected):
                self.assert_code(
                    expected, lambda body=body: parse_vllm_metrics(body)
                )

    def test_histogram_pair_and_delta_inconsistency_fail_closed(self) -> None:
        before = parse_vllm_metrics(
            "vllm:time_to_first_token_seconds_sum 1\n"
            "vllm:time_to_first_token_seconds_count 2"
        )
        incomplete = parse_vllm_metrics(
            "vllm:time_to_first_token_seconds_sum 2"
        )
        self.assert_code(
            "metrics_histogram_pair_incomplete",
            lambda: derive_metrics_window(before, incomplete, 1),
        )
        bad_delta = parse_vllm_metrics(
            "vllm:time_to_first_token_seconds_sum 2\n"
            "vllm:time_to_first_token_seconds_count 2"
        )
        self.assert_code(
            "metrics_histogram_delta_mismatch",
            lambda: derive_metrics_window(before, bad_delta, 1),
        )
        per_series_before = parse_vllm_metrics(
            'vllm:time_to_first_token_seconds_sum{engine="a"} 1\n'
            'vllm:time_to_first_token_seconds_count{engine="a"} 1\n'
            'vllm:time_to_first_token_seconds_sum{engine="b"} 1\n'
            'vllm:time_to_first_token_seconds_count{engine="b"} 1'
        )
        per_series_after = parse_vllm_metrics(
            'vllm:time_to_first_token_seconds_sum{engine="a"} 2\n'
            'vllm:time_to_first_token_seconds_count{engine="a"} 1\n'
            'vllm:time_to_first_token_seconds_sum{engine="b"} 1\n'
            'vllm:time_to_first_token_seconds_count{engine="b"} 2'
        )
        self.assert_code(
            "metrics_histogram_delta_mismatch",
            lambda: derive_metrics_window(
                per_series_before, per_series_after, 1
            ),
        )

    def test_invalid_elapsed_and_snapshot_types_fail_closed(self) -> None:
        empty = parse_vllm_metrics("")
        for elapsed in (
            0,
            -1,
            math.nan,
            math.inf,
            10**400,
            True,
            "1",
        ):
            with self.subTest(elapsed=elapsed):
                self.assert_code(
                    "metrics_invalid_elapsed",
                    lambda elapsed=elapsed: derive_metrics_window(
                        empty, empty, elapsed  # type: ignore[arg-type]
                    ),
                )
        self.assert_code(
            "metrics_invalid_snapshot",
            lambda: derive_metrics_window({}, empty, 1),  # type: ignore[arg-type]
        )
        forged = replace(empty, recognized_samples=1)
        self.assert_code(
            "metrics_invalid_snapshot",
            lambda: derive_metrics_window(forged, forged, 1),
        )
        self.assert_code(
            "metrics_invalid_snapshot", forged.public_summary
        )
        mutated = parse_vllm_metrics("vllm:generation_tokens_total 1")
        object.__setattr__(mutated._families[0][1][0], "value", 1e100)
        self.assert_code(
            "metrics_invalid_snapshot", mutated.public_summary
        )
        private = "https://private.example/api_key=secret"
        equal_secret = _EqualSecret(private)
        forged_family = replace(
            mutated, available_families=(equal_secret,)
        )
        with self.assertRaises(MetricsCollectionError) as raised:
            forged_family.public_summary()
        self.assertEqual(raised.exception.code, "metrics_invalid_snapshot")
        self.assertNotIn(private, str(raised.exception))
        self.assert_code(
            "metrics_invalid_observations",
            lambda: derive_metrics_window(
                empty,
                empty,
                1,
                observations=(item for item in (empty,)),  # type: ignore[arg-type]
            ),
        )
        self.assert_code(
            "metrics_too_many_observations",
            lambda: derive_metrics_window(
                empty,
                empty,
                1,
                observations=[empty] * (MAX_OBSERVATIONS + 1),
            ),
        )

    def test_window_has_a_cumulative_recognized_sample_cap(self) -> None:
        samples_per_snapshot = MAX_RECOGNIZED_SAMPLES
        snapshot = parse_vllm_metrics(
            "\n".join(
                _line(
                    "vllm:generation_tokens_total",
                    1,
                    series=str(index),
                )
                for index in range(samples_per_snapshot)
            )
        )
        total_snapshots = (
            MAX_WINDOW_RECOGNIZED_SAMPLES // samples_per_snapshot
        ) + 1
        self.assert_code(
            "metrics_window_too_large",
            lambda: derive_metrics_window(
                snapshot,
                snapshot,
                1,
                observations=[snapshot] * (total_snapshots - 2),
            ),
        )


class CollectorTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_constructor_is_offline_and_url_policy_is_fail_closed(self) -> None:
        with patch.object(
            httpx.AsyncClient, "__init__", side_effect=AssertionError("network")
        ):
            PrometheusMetricsCollector("https://metrics.example/metrics")

        accepted = (
            "https://metrics.example/custom/prometheus",
            "http://127.0.0.1:8000/metrics",
            "http://[::1]:8000/metrics",
            "http://localhost:8000/metrics",
        )
        for value in accepted:
            with self.subTest(value=value):
                PrometheusMetricsCollector(value)

        rejected = (
            "",
            "metrics.example/metrics",
            "http://metrics.example/metrics",
            "http://0.0.0.0:8000/metrics",
            "https://user:password@metrics.example/metrics",
            "https://metrics.example/metrics?secret=value",
            "https://metrics.example/metrics?",
            "https://metrics.example/metrics#fragment",
            "https://metrics.example/metrics#",
            "https://*.example/metrics",
            "https://metrics.example/../private",
            "https://metrics.example",
            " https://metrics.example/metrics",
            "https://metrics.example/metrics\nprivate",
            "https://metrics.example/private path",
            "https://metrics.example/%0d%0aprivate",
            "https://metrics.example/%2e%2e/private",
            "https://metrics.example/%252e%252e/private",
            "https://metrics.example/%10private",
            "https://metrics.example/%1fprivate",
            "https://bad_host.example/metrics",
            "https:\\metrics.example\\metrics",
        )
        for value in rejected:
            with self.subTest(value=value):
                with self.assertRaises(MetricsCollectionError) as raised:
                    PrometheusMetricsCollector(value)
                self.assertIn(
                    raised.exception.code,
                    {"metrics_invalid_url", "metrics_insecure_url"},
                )
                if value:
                    self.assertNotIn(value, str(raised.exception))

    async def test_scrape_is_exact_bounded_unauthenticated_get(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/plain; version=0.0.4; charset=utf-8"
                },
                stream=httpx.ByteStream(
                    b"vllm:generation_tokens_total 7\n"
                ),
            )

        collector = PrometheusMetricsCollector(
            "https://metrics.example/custom",
            transport=httpx.MockTransport(handler),
        )
        snapshot = await collector.snapshot()
        self.assertEqual(snapshot.recognized_samples, 1)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(str(request.url), "https://metrics.example/custom")
        self.assertEqual(
            request.headers["accept"], "text/plain; version=0.0.4"
        )
        self.assertEqual(request.headers["accept-encoding"], "identity")
        self.assertNotIn("authorization", request.headers)
        self.assertNotIn("cookie", request.headers)

    async def test_scrape_failures_use_fixed_nonreflective_codes(self) -> None:
        private = "private-location.example/private-token"
        cases: tuple[tuple[str, object], ...] = (
            (
                "metrics_redirect_rejected",
                httpx.Response(302, headers={"Location": f"https://{private}"}),
            ),
            (
                "metrics_http_status",
                httpx.Response(503, text=private),
            ),
            (
                "metrics_content_type_rejected",
                httpx.Response(200, headers={"Content-Type": "text/html"}),
            ),
            (
                "metrics_content_type_rejected",
                httpx.Response(
                    200,
                    headers={"Content-Type": "application/openmetrics-text"},
                ),
            ),
            (
                "metrics_content_encoding_rejected",
                httpx.Response(
                    200,
                    headers={
                        "Content-Type": "text/plain",
                        "Content-Encoding": "gzip",
                    },
                    stream=httpx.ByteStream(b"not-gzip"),
                ),
            ),
            (
                "metrics_invalid_content_length",
                httpx.Response(
                    200,
                    headers={
                        "Content-Type": "text/plain",
                        "Content-Length": "private",
                    },
                ),
            ),
            (
                "metrics_response_too_large",
                httpx.Response(
                    200,
                    headers={"Content-Type": "text/plain"},
                    stream=httpx.ByteStream(
                        b"x" * (MAX_RESPONSE_BYTES + 1)
                    ),
                ),
            ),
        )
        for expected, response in cases:
            with self.subTest(expected=expected):
                collector = PrometheusMetricsCollector(
                    "https://metrics.example/metrics",
                    transport=httpx.MockTransport(
                        lambda _: response  # type: ignore[arg-type]
                    ),
                )
                with self.assertRaises(MetricsCollectionError) as raised:
                    await collector.snapshot()
                self.assertEqual(str(raised.exception), expected)
                self.assertNotIn(private, str(raised.exception))

    async def test_timeout_and_transport_errors_are_sanitized(self) -> None:
        errors = (
            ("metrics_timeout", httpx.ReadTimeout("private timeout")),
            (
                "metrics_transport_error",
                httpx.ConnectError("private endpoint failure"),
            ),
        )
        for expected, error in errors:
            with self.subTest(expected=expected):
                def handler(_: httpx.Request, error: Exception = error) -> object:
                    raise error

                collector = PrometheusMetricsCollector(
                    "https://metrics.example/metrics",
                    transport=httpx.MockTransport(handler),
                )
                with self.assertRaises(MetricsCollectionError) as raised:
                    await collector.snapshot()
                self.assertEqual(str(raised.exception), expected)
                self.assertNotIn("private", str(raised.exception))

    async def test_entire_scrape_has_one_deadline(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain"},
                stream=_DelayedStream(),
            )

        collector = PrometheusMetricsCollector(
            "https://metrics.example/metrics",
            transport=httpx.MockTransport(handler),
        )
        with patch(
            "throttle.server_metrics.SCRAPE_TIMEOUT_SECONDS", 0.01
        ):
            with self.assertRaises(MetricsCollectionError) as raised:
                await collector.snapshot()
        self.assertEqual(raised.exception.code, "metrics_timeout")

    async def test_chunked_body_cap_does_not_rely_on_content_length(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain"},
                stream=_ChunkedOversizeStream(),
            )

        collector = PrometheusMetricsCollector(
            "https://metrics.example/metrics",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(MetricsCollectionError) as raised:
            await collector.snapshot()
        self.assertEqual(raised.exception.code, "metrics_response_too_large")


if __name__ == "__main__":
    unittest.main()
