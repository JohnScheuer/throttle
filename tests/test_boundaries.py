from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
import unicodedata
from dataclasses import replace
from pathlib import Path
from unittest import mock

from throttle.benchmark import validate_config
from throttle.compare import (
    MAX_REPORT_DEPTH,
    MAX_REPORT_NODES,
    MAX_REPORT_STRING_BYTES,
    MAX_REPORT_STRING_LENGTH,
    MAX_SAFE_NUMERIC_MAGNITUDE,
    ComparisonInputError,
    compare_reports,
    load_report,
    validate_report_structure,
)
from throttle.golden import validate_golden_sequence
from throttle.provenance import (
    is_immutable_artifact_digest,
    is_safe_artifact_reference,
    is_safe_public_metadata,
)

from test_benchmark import (
    _golden_sequence,
    _run_config,
    _saved_report,
    _set_metal_runtime,
)


def _control_at_offsets(character: str) -> tuple[str, str, str]:
    return (
        character + "private-runtime",
        "private" + character + "runtime",
        "private-runtime" + character,
    )


def _credential_at_offsets(fragment: str) -> tuple[str, str, str]:
    # Punctuation and whitespace preserve a neutral token boundary while moving
    # the credential-bearing fragment through the public value.
    return (
        fragment + "/details",
        "runtime " + fragment + "/details",
        "runtime metadata " + fragment,
    )


def _metadata_at_offsets(fragment: str) -> tuple[str, str, str]:
    return (
        fragment,
        "runtime " + fragment,
        "runtime metadata " + fragment + " details",
    )


def _nested_list(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return value


def _node_list(nodes: int) -> list[object]:
    if nodes < 1:
        raise ValueError("a JSON tree must contain at least its root node")
    return [None] * (nodes - 1)


def _render(value: object) -> str:
    if isinstance(value, BaseException):
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _assert_not_reflected(
    testcase: unittest.TestCase, payload: str, value: object
) -> None:
    rendered = _render(value)
    escaped = json.dumps(payload, ensure_ascii=True)[1:-1]
    testcase.assertNotIn(payload, rendered)
    testcase.assertNotIn(escaped, rendered)


class MetadataBoundaryTests(unittest.TestCase):
    def test_unicode_controls_and_spoofing_characters_fail_at_every_offset(
        self,
    ) -> None:
        unsafe_characters = (
            "\x00",  # Cc
            "\x1b",
            "\x7f",
            "\u0085",
            "\u200b",  # Cf
            "\ufeff",
            "\u202a",  # bidi embedding/override controls
            "\u202e",
            "\u2066",  # bidi isolates
            "\u2069",
            "\ud800",  # Cs
            "\udfff",
            "\u2028",  # Zl
            "\u2029",  # Zp
        )
        for character in unsafe_characters:
            for offset, payload in enumerate(_control_at_offsets(character)):
                with self.subTest(codepoint=f"U+{ord(character):04X}", offset=offset):
                    self.assertFalse(is_safe_public_metadata(payload))
                    config = replace(_run_config(), host_os_version=payload)
                    with self.assertRaises(ValueError) as raised:
                        validate_config(config, for_traffic=False)
                    _assert_not_reflected(self, payload, raised.exception)

    def test_benign_unicode_remains_public(self) -> None:
        for value in (
            "Apple M3 Pro – München",
            "AMD EPYC 9654 — 東京",
            "Café Ω accelerator",
            "מערכת בדיקה בטוחה",
            "Ｍetal accelerator １２３",
            "① measured condition",
            "ﬂow-control runtime",
        ):
            with self.subTest(value=value):
                self.assertTrue(is_safe_public_metadata(value))
                validate_config(
                    replace(_run_config(), host_os_version=value),
                    for_traffic=False,
                )

    def test_nfkc_credential_url_userinfo_and_path_bypasses_fail_closed(
        self,
    ) -> None:
        attacks = (
            "ｓｋ＿abcdefghijklmno",
            "ｇｈｐ＿abcdefghijklmnop",
            "ｈｔｔｐｓ：／／private.example/runtime",
            "ａｌｉｃｅ：ｈｕｎｔｅｒ２＠private.example",
            "ｆｉｌｅ：／Users／private／config",
            "／Users／private／runtime",
            "．．／private／runtime",
            "Ｃ：＼Users＼private＼runtime",
        )
        for attack in attacks:
            normalized = unicodedata.normalize("NFKC", attack)
            self.assertNotEqual(normalized, attack)
            for offset, payload in enumerate(_metadata_at_offsets(attack)):
                with self.subTest(normalized=normalized[:20], offset=offset):
                    self.assertFalse(is_safe_public_metadata(payload))
                    with self.assertRaises(ValueError) as raised:
                        validate_config(
                            replace(_run_config(), host_os_version=payload),
                            for_traffic=False,
                        )
                    _assert_not_reflected(self, payload, raised.exception)

    def test_credentials_are_rejected_at_deterministic_offsets_without_echo(
        self,
    ) -> None:
        fragments = (
            "token=PRIVATE_TOKEN_VALUE",
            "ACCESS-TOKEN:PRIVATE_TOKEN_VALUE",
            "api_key=PRIVATE_API_VALUE",
            "password=PRIVATE_PASSWORD_VALUE",
            "authorization: Bearer PRIVATE_AUTH_VALUE",
            "ghp_abcdefghijklmnop",
            "github_pat_abcdefghijklmnop",
            "AKIA" + "A" * 16,
            "xoxb-1234567890abcdef",
            "hf_abcdefghijklmnop",
            "eyJabc.eyJdef.signature",
            "alice:hunter2@private.example",
        )
        for fragment in fragments:
            for offset, payload in enumerate(_credential_at_offsets(fragment)):
                with self.subTest(fragment=fragment[:12], offset=offset):
                    self.assertFalse(is_safe_public_metadata(payload))
                    config = replace(_run_config(), host_os_version=payload)
                    with self.assertRaises(ValueError) as raised:
                        validate_config(config, for_traffic=False)
                    _assert_not_reflected(self, payload, raised.exception)

    def test_long_userinfo_components_fail_saved_boundaries_without_echo(
        self,
    ) -> None:
        payloads = (
            "alice:" + "p" * 257 + "@private.example",
            "u" * 129 + ":hunter2@private.example",
            "alice:hunter2@" + "h" * 513 + ".example",
        )
        for payload in payloads:
            with self.subTest(path="public", length=len(payload)):
                self.assertFalse(
                    is_safe_public_metadata(
                        payload,
                        max_length=MAX_REPORT_STRING_LENGTH,
                    )
                )
                self.assertEqual(
                    validate_report_structure({"ignored_extra": payload}),
                    "report_structure_unsafe_string",
                )

            with self.subTest(path="compare", length=len(payload)):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                baseline["ignored_extra"] = payload
                result = compare_reports(baseline, candidate)
                self.assertEqual(
                    result["compatibility"]["reasons"],
                    ["baseline_report_structure_unsafe_string"],
                )
                _assert_not_reflected(self, payload, result)

            with self.subTest(path="golden", length=len(payload)):
                reports = _golden_sequence()
                reports[0]["ignored_extra"] = payload
                result = validate_golden_sequence(reports)
                self.assertEqual(
                    result["eligibility_reasons"],
                    ["run_1_report_structure_unsafe_string"],
                )
                self.assertIsNone(result["run_fingerprints"][0])
                _assert_not_reflected(self, payload, result)

    def test_digest_suffix_does_not_mask_non_oci_userinfo(self) -> None:
        digest = "sha256:" + "a" * 64
        for payload in (
            "alice%corp:hunter2@" + digest,
            "alice$corp:hunter2@" + digest,
        ):
            with self.subTest(path="generated", payload=payload[:20]):
                self.assertFalse(is_safe_public_metadata(payload))
                config = replace(_run_config(), host_os_version=payload)
                with self.assertRaises(ValueError) as raised:
                    validate_config(config, for_traffic=False)
                _assert_not_reflected(self, payload, raised.exception)

            with self.subTest(path="compare", payload=payload[:20]):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_metal_runtime(report)
                runtime = baseline["manifest"]["runtime"]  # type: ignore[index]
                runtime["software_environment_digest"] = payload  # type: ignore[index]

                result = compare_reports(baseline, candidate)

                self.assertEqual(
                    result["compatibility"]["reasons"],
                    ["baseline_unsafe_runtime_metadata"],
                )
                _assert_not_reflected(self, payload, result)

            with self.subTest(path="golden", payload=payload[:20]):
                reports = _golden_sequence()
                for report in reports:
                    _set_metal_runtime(report)
                runtime = reports[0]["manifest"]["runtime"]  # type: ignore[index]
                runtime["software_environment_digest"] = payload  # type: ignore[index]

                result = validate_golden_sequence(reports)

                self.assertEqual(
                    result["eligibility_reasons"],
                    ["run_1_unsafe_runtime_metadata"],
                )
                self.assertIsNone(result["run_fingerprints"][0])
                _assert_not_reflected(self, payload, result)

    def test_engine_flag_values_share_the_public_metadata_boundary(self) -> None:
        for payload in (
            "runtime token=PRIVATE_ENGINE_TOKEN",
            "runtime ghp_abcdefghijklmnop",
            "runtime file:/Users/private/config",
            "runtime\u202esecret",
        ):
            with self.subTest(payload=payload):
                config = replace(
                    _run_config(), engine_flags=(("max-num-batched-tokens", payload),)
                )
                with self.assertRaises(ValueError) as raised:
                    validate_config(config, for_traffic=False)
                _assert_not_reflected(self, payload, raised.exception)

        for value in (
            "4096",
            "tokenizer=v2",
            "max-num-batched-tokens=4096",
            "sentencepiece-token-counting",
        ):
            with self.subTest(nonsecret=value):
                self.assertTrue(is_safe_public_metadata(value))
                validate_config(
                    replace(
                        _run_config(),
                        engine_flags=(("max-num-batched-tokens", value),),
                    ),
                    for_traffic=False,
                )

    def test_engine_flag_names_reject_secret_markers_without_reflection(self) -> None:
        hostile_names = (
            "private_endpoint_secret",
            "authorization_header",
            "ghp_abcdefghijklmnop",
            "sk_abcdefghijklmnop",
            "AKIA" + "A" * 16,
            "ｇｈｐ＿abcdefghijklmnop",
        )
        for name in hostile_names:
            with self.subTest(name=name[:20]):
                config = replace(_run_config(), engine_flags=((name, "4096"),))
                with self.assertRaises(ValueError) as raised:
                    validate_config(config, for_traffic=False)
                _assert_not_reflected(self, name, raised.exception)

        for name in (
            "max_num_batched_tokens",
            "tokenizer",
            "sentencepiece_token_counting",
        ):
            with self.subTest(nonsecret=name):
                validate_config(
                    replace(_run_config(), engine_flags=((name, "4096"),)),
                    for_traffic=False,
                )

    def test_paths_urls_and_traversal_are_rejected_at_public_boundary(self) -> None:
        unsafe_values = (
            "/Users/private/runtime",
            "~/private/runtime",
            "C:\\Users\\private\\runtime",
            "\\\\private-server\\share",
            "../private/runtime",
            "runtime/../private",
            "runtime\\..\\private",
            "https://private.example/runtime",
            "runtime https://private.example/runtime",
            "file:/Users/private/runtime",
            "runtime file:/Users/private/runtime",
            "FILE://private.example/runtime",
            "s3:private-bucket/runtime",
            "runtime ssh:private.example",
            "mailto:private@example.com",
            "cwd:/Users/private",
            "container:/workspace/model",
            "path:C:\\Users\\private",
            "runtime:/root/.cache",
        )
        for value in unsafe_values:
            with self.subTest(value=value):
                self.assertFalse(is_safe_public_metadata(value))
                with self.assertRaises(ValueError) as raised:
                    validate_config(
                        replace(_run_config(), host_os_version=value),
                        for_traffic=False,
                    )
                _assert_not_reflected(self, value, raised.exception)

    def test_nfkc_expansion_is_bounded_and_charged_to_report_budget(
        self,
    ) -> None:
        pathological = "\ufdfa" * 900
        self.assertGreater(
            len(unicodedata.normalize("NFKC", pathological)),
            len(pathological) * 4,
        )
        self.assertFalse(
            is_safe_public_metadata(pathological, max_length=16_384)
        )
        self.assertEqual(
            validate_report_structure({"value": pathological}),
            "report_structure_unsafe_string",
        )

        bounded = "\u337f"
        normalized = unicodedata.normalize("NFKC", bounded)
        self.assertEqual(normalized, "株式会社")
        self.assertTrue(is_safe_public_metadata(bounded, max_length=16_384))
        with mock.patch("throttle.compare.MAX_REPORT_STRING_BYTES", 14):
            self.assertIsNone(validate_report_structure({"k": bounded}))
        with mock.patch("throttle.compare.MAX_REPORT_STRING_BYTES", 12):
            self.assertEqual(
                validate_report_structure({"k": bounded}),
                "report_structure_too_large",
            )


class ArtifactAndReportBoundaryTests(unittest.TestCase):
    def test_artifact_digest_shapes_preserve_oci_and_separate_immutability(
        self,
    ) -> None:
        lower64 = "a" * 64
        cases = (
            ("sha256:" + lower64, True, True),
            ("image@sha256:" + lower64, True, True),
            ("ubuntu:24.04@sha256:" + lower64, True, True),
            (
                "registry.example:5000/team/image:tag@sha256:" + lower64,
                True,
                True,
            ),
            (
                "registry.example:5000/repo/image:tag@sha256:" + lower64,
                True,
                True,
            ),
            ("registry/image:tag@sha256:" + lower64, True, True),
            ("image:latest", True, False),
            ("image@sha512:" + "b" * 128, True, False),
            ("image@sha256:" + "A" * 64, True, False),
            ("image@SHA256:" + lower64, True, False),
            ("image@sha256:" + "a" * 63, True, False),
            ("image@sha256:" + "a" * 65, True, False),
            ("image@x:" + "a" * 15, False, False),
            ("image@x:" + "a" * 16, True, False),
            ("image@@sha256:" + lower64, False, False),
            ("user:password@sha256:" + lower64, False, False),
            ("alice%corp:hunter2@sha256:" + lower64, False, False),
            ("alice$corp:hunter2@sha256:" + lower64, False, False),
            ("image@sha256:" + "g" * 64, False, False),
            ("image@sha256:" + lower64 + " ", False, False),
            ("../image@sha256:" + lower64, False, False),
            ("https://private.example/image@sha256:" + lower64, False, False),
        )
        for value, safe, immutable in cases:
            with self.subTest(value=value[:40], length=len(value)):
                self.assertEqual(is_safe_artifact_reference(value), safe)
                self.assertEqual(is_immutable_artifact_digest(value), immutable)

    def test_mutable_and_noncanonical_digests_remain_descriptive_only(self) -> None:
        for reference in (
            "image:latest",
            "image@sha512:" + "b" * 128,
            "image@sha256:" + "A" * 64,
            "image@sha256:" + "a" * 63,
            "image@sha256:" + "a" * 65,
        ):
            with self.subTest(reference=reference[:32]):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    runtime = report["manifest"]["runtime"]  # type: ignore[index]
                    runtime["image_digest"] = reference  # type: ignore[index]

                result = compare_reports(baseline, candidate)

                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["compatibility"]["compatible"])
                self.assertFalse(result["decision_eligible"])
                self.assertIsNotNone(result["descriptive_statistical_outcome"])
                self.assertIn(
                    "immutable_image_digest_required",
                    result["decision_ineligible_reasons"],
                )

    def test_nested_and_escaped_equivalent_duplicate_keys_are_sanitized(self) -> None:
        secret = "PRIVATE_DUPLICATE_JSON_PAYLOAD"
        payloads = (
            '{"manifest":{"runtime":{"gpu":"safe","gpu":"'
            + secret
            + '"}}}',
            '{"manifest":{"runtime":{"gpu":"safe","g\\u0070u":"'
            + secret
            + '"}}}',
            '{"conditions":[{"metrics":{"value":1,"value":"'
            + secret
            + '"}}]}',
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, payload in enumerate(payloads):
                with self.subTest(index=index):
                    path = Path(temp_dir) / f"duplicate-{index}.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ComparisonInputError) as raised:
                        load_report(path)
                    self.assertEqual(
                        str(raised.exception),
                        "saved report is unreadable or not valid JSON",
                    )
                    _assert_not_reflected(self, secret, raised.exception)

    def test_alias_mutations_fail_closed_without_payload_reflection(self) -> None:
        cases = (
            (
                "gpu",
                "PRIVATE_CONTRADICTORY_ACCELERATOR",
                "runtime_aliases_do_not_reconcile",
            ),
            (
                "accelerator_fingerprint_sha256",
                "1" * 64,
                "runtime_aliases_do_not_reconcile",
            ),
            (
                "gpu_fingerprint_supplied",
                False,
                "runtime_aliases_do_not_reconcile",
            ),
            (
                "gpu_fingerprint_sha256",
                "PRIVATE_MALFORMED_DIGEST",
                "invalid_manifest_digest",
            ),
        )
        for field, payload, reason in cases:
            with self.subTest(field=field):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_metal_runtime(report)
                baseline["manifest"]["runtime"][field] = payload  # type: ignore[index]

                result = compare_reports(baseline, candidate)

                self.assertEqual(result["status"], "incompatible")
                self.assertFalse(result["decision_eligible"])
                self.assertIn(
                    "baseline_" + reason, result["compatibility"]["reasons"]
                )
                if isinstance(payload, str):
                    _assert_not_reflected(self, payload, result)

    def test_saved_credentials_are_rejected_without_compare_or_golden_echo(
        self,
    ) -> None:
        for payload in (
            "runtime token=PRIVATE_SAVED_ASSIGNMENT",
            "runtime ghp_abcdefghijklmnop",
            "runtime alice:hunter2@private.example",
        ):
            with self.subTest(path="compare", payload=payload[:20]):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_metal_runtime(report)
                runtime = baseline["manifest"]["runtime"]  # type: ignore[index]
                runtime["gpu"] = payload  # type: ignore[index]
                runtime["accelerator"] = payload  # type: ignore[index]

                result = compare_reports(baseline, candidate)

                self.assertEqual(
                    result["compatibility"]["reasons"],
                    ["baseline_unsafe_runtime_metadata"],
                )
                _assert_not_reflected(self, payload, result)

            with self.subTest(path="golden", payload=payload[:20]):
                reports = _golden_sequence()
                for report in reports:
                    _set_metal_runtime(report)
                runtime = reports[0]["manifest"]["runtime"]  # type: ignore[index]
                runtime["gpu"] = payload  # type: ignore[index]
                runtime["accelerator"] = payload  # type: ignore[index]

                result = validate_golden_sequence(reports)

                self.assertIn(
                    "run_1_unsafe_runtime_metadata",
                    result["eligibility_reasons"],
                )
                self.assertIsNone(result["run_fingerprints"][0])
                _assert_not_reflected(self, payload, result)

    def test_saved_engine_flag_names_fail_compare_and_golden_without_echo(
        self,
    ) -> None:
        hostile_names = (
            "private_endpoint_secret",
            "authorization_header",
            "ghp_abcdefghijklmnop",
            "sk_abcdefghijklmnop",
            "AKIA" + "A" * 16,
            "ｇｈｐ＿abcdefghijklmnop",
        )
        for name in hostile_names:
            with self.subTest(path="compare", name=name[:20]):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                engine = baseline["manifest"]["engine"]  # type: ignore[index]
                flags = engine["effective_flags"]  # type: ignore[index]
                flags[name] = "4096"  # type: ignore[index]

                result = compare_reports(baseline, candidate)

                self.assertEqual(
                    result["compatibility"]["reasons"],
                    ["baseline_invalid_engine_flag_manifest"],
                )
                _assert_not_reflected(self, name, result)

            with self.subTest(path="golden", name=name[:20]):
                reports = _golden_sequence()
                engine = reports[0]["manifest"]["engine"]  # type: ignore[index]
                flags = engine["effective_flags"]  # type: ignore[index]
                flags[name] = "4096"  # type: ignore[index]

                result = validate_golden_sequence(reports)

                self.assertEqual(
                    result["eligibility_reasons"],
                    ["run_1_invalid_engine_flag_manifest"],
                )
                self.assertIsNone(result["run_fingerprints"][0])
                _assert_not_reflected(self, name, result)

    def test_ignored_safe_extra_does_not_change_golden_fingerprint(self) -> None:
        reports = _golden_sequence()
        reference = validate_golden_sequence(reports)
        extended = copy.deepcopy(reports)
        extended[0]["future_extension"] = {
            "label": "Café benign extension",
            "revision": 1,
        }
        runtime = extended[0]["manifest"]["runtime"]  # type: ignore[index]
        runtime[  # type: ignore[index]
            "future_runtime_label"
        ] = "benign runtime extension"
        metrics = extended[0]["conditions"][0]["metrics"]  # type: ignore[index]
        metrics["future_metric_note"] = "benign metric extension"  # type: ignore[index]

        result = validate_golden_sequence(extended)

        self.assertEqual(result["status"], reference["status"])
        self.assertEqual(
            result["golden_protocol_eligible"],
            reference["golden_protocol_eligible"],
        )
        self.assertEqual(
            result["eligibility_reasons"], reference["eligibility_reasons"]
        )
        self.assertEqual(result["run_fingerprints"], reference["run_fingerprints"])
        self.assertIsNotNone(result["run_fingerprints"][0])

        unsafe = copy.deepcopy(reports)
        payload = "runtime ghp_PRIVATE_IGNORED_EXTENSION"
        unsafe[0]["future_extension"] = payload
        rejected = validate_golden_sequence(unsafe)
        self.assertIn(
            "run_1_report_structure_unsafe_string",
            rejected["eligibility_reasons"],
        )
        self.assertIsNone(rejected["run_fingerprints"][0])
        _assert_not_reflected(self, payload, rejected)

    def test_safe_unconsumed_known_metrics_do_not_change_golden_fingerprint(
        self,
    ) -> None:
        reports = _golden_sequence()
        reference = validate_golden_sequence(reports)
        self.assertIsNotNone(reference["run_fingerprints"][0])

        for field in ("e2e_p50", "cost_metric_basis", "prompt_tokens"):
            with self.subTest(field=field):
                changed = copy.deepcopy(reports)
                condition = changed[0]["conditions"][0]
                metric_sets = [condition["metrics"]]
                metric_sets.extend(block["metrics"] for block in condition["blocks"])
                if field == "e2e_p50":
                    for metrics in metric_sets:
                        metrics["e2e_latency_ms"]["p50"] = 9.0
                elif field == "cost_metric_basis":
                    for metrics in metric_sets:
                        metrics[field] = "benign alternate descriptive basis"
                else:
                    for metrics in metric_sets:
                        metrics[field] += 1

                result = validate_golden_sequence(changed)

                self.assertEqual(result["status"], reference["status"])
                self.assertEqual(
                    result["golden_protocol_eligible"],
                    reference["golden_protocol_eligible"],
                )
                self.assertEqual(
                    result["eligibility_reasons"],
                    reference["eligibility_reasons"],
                )
                self.assertEqual(
                    result["run_fingerprints"],
                    reference["run_fingerprints"],
                )

    def test_consumed_known_metrics_change_or_suppress_golden_fingerprint(
        self,
    ) -> None:
        reports = _golden_sequence()
        reference = validate_golden_sequence(reports)
        self.assertIsNotNone(reference["run_fingerprints"][0])

        for field, reason in (
            (
                "repeated_block_p95_high",
                "repeated_block_slo_interval_does_not_reconcile",
            ),
            ("completion_tokens", "condition_tokens_do_not_reconcile"),
            ("output_tokens_per_second", "condition_throughput_does_not_reconcile"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(reports)
                metrics = changed[0]["conditions"][0]["metrics"]
                if field == "repeated_block_p95_high":
                    repeated = metrics["e2e_latency_ms"][
                        "p95_repeated_block_ci"
                    ]
                    repeated["high"] += 1.0
                else:
                    metrics[field] += 1

                result = validate_golden_sequence(changed)

                self.assertEqual(
                    result["eligibility_reasons"], ["run_1_" + reason]
                )
                self.assertIsNone(result["run_fingerprints"][0])
                self.assertNotEqual(
                    result["run_fingerprints"],
                    reference["run_fingerprints"],
                )

    def test_structure_then_schema_then_string_reason_precedence(self) -> None:
        malformed_digest = "PRIVATE_SCHEMA_DIGEST_PAYLOAD"
        cases = (
            (
                math.nan,
                "report_structure_nonfinite_number",
            ),
            (
                "runtime ghp_PRIVATE_IGNORED_PAYLOAD",
                "invalid_manifest_digest",
            ),
        )
        for extra, reason in cases:
            with self.subTest(path="compare", reason=reason):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_metal_runtime(report)
                baseline["manifest"]["runtime"][  # type: ignore[index]
                    "gpu_fingerprint_sha256"
                ] = malformed_digest
                baseline["ignored_extra"] = extra

                result = compare_reports(baseline, candidate)

                self.assertEqual(
                    result["compatibility"]["reasons"], ["baseline_" + reason]
                )
                _assert_not_reflected(self, malformed_digest, result)
                if isinstance(extra, str):
                    _assert_not_reflected(self, extra, result)

            with self.subTest(path="golden", reason=reason):
                reports = _golden_sequence()
                reports[0]["manifest"]["runtime"][  # type: ignore[index]
                    "gpu_fingerprint_sha256"
                ] = malformed_digest
                reports[0]["ignored_extra"] = extra

                result = validate_golden_sequence(reports)

                self.assertEqual(
                    result["eligibility_reasons"], ["run_1_" + reason]
                )
                self.assertIsNone(result["run_fingerprints"][0])
                _assert_not_reflected(self, malformed_digest, result)
                if isinstance(extra, str):
                    _assert_not_reflected(self, extra, result)


class StructureBoundaryTests(unittest.TestCase):
    def test_numeric_and_json_type_boundaries_have_fixed_codes(self) -> None:
        maximum = MAX_SAFE_NUMERIC_MAGNITUDE
        accepted = (None, True, False, -maximum, maximum, 0.0, 1.5)
        for value in accepted:
            with self.subTest(accepted=repr(value)):
                self.assertIsNone(validate_report_structure({"value": value}))

        rejected = (
            (math.nan, "report_structure_nonfinite_number"),
            (math.inf, "report_structure_nonfinite_number"),
            (-math.inf, "report_structure_nonfinite_number"),
            (maximum + 1, "report_structure_oversize_number"),
            (-(maximum + 1), "report_structure_oversize_number"),
            (
                math.nextafter(float(maximum), math.inf),
                "report_structure_oversize_number",
            ),
            ({1, 2}, "report_structure_non_json_type"),
            (b"private", "report_structure_non_json_type"),
            (object(), "report_structure_non_json_type"),
            ("unsafe\ud800string", "report_structure_unsafe_string"),
        )
        for value, reason in rejected:
            with self.subTest(reason=reason, value_type=type(value).__name__):
                self.assertEqual(validate_report_structure({"value": value}), reason)

    def test_compare_rejects_numeric_structure_without_nonfinite_output(self) -> None:
        cases = (
            (math.nan, "report_structure_nonfinite_number"),
            (MAX_SAFE_NUMERIC_MAGNITUDE + 1, "report_structure_oversize_number"),
            ({"not-json"}, "report_structure_non_json_type"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                baseline["untrusted_extra"] = value

                result = compare_reports(baseline, candidate)

                self.assertEqual(result["status"], "incompatible")
                self.assertEqual(
                    result["compatibility"]["reasons"], ["baseline_" + reason]
                )
                json.dumps(result, allow_nan=False)

    def test_explicit_depth_and_node_boundaries(self) -> None:
        self.assertEqual(MAX_REPORT_DEPTH, 64)
        self.assertEqual(MAX_REPORT_NODES, 100_000)
        self.assertIsNone(validate_report_structure(_nested_list(MAX_REPORT_DEPTH)))
        self.assertEqual(
            validate_report_structure(_nested_list(MAX_REPORT_DEPTH + 1)),
            "report_structure_too_deep",
        )
        self.assertIsNone(validate_report_structure(_node_list(MAX_REPORT_NODES)))
        self.assertEqual(
            validate_report_structure(_node_list(MAX_REPORT_NODES + 1)),
            "report_structure_too_large",
        )

    def test_aggregate_string_budget_counts_keys_and_values_at_cap(self) -> None:
        self.assertEqual(MAX_REPORT_STRING_BYTES, 20_000_000)
        # Shrink only the limit so this exact edge exercises UTF-8 accounting
        # without making a security regression test allocate or scan 20 MB.
        values = ["é" * 7]  # Fourteen UTF-8 bytes.
        with mock.patch("throttle.compare.MAX_REPORT_STRING_BYTES", 15):
            self.assertIsNone(validate_report_structure({"k": values}))
            self.assertEqual(
                validate_report_structure({"kk": values}),
                "report_structure_too_large",
            )

    def test_long_benign_metadata_at_individual_limit_remains_bounded(self) -> None:
        sixteen_kib_unicode = "é " * 5_461 + "a"
        maximum_ascii = "a" * MAX_REPORT_STRING_LENGTH
        maximum_punctuation = "x-" * (MAX_REPORT_STRING_LENGTH // 2)
        self.assertEqual(len(sixteen_kib_unicode.encode("utf-8")), 16_384)
        for value in (sixteen_kib_unicode, maximum_ascii, maximum_punctuation):
            with self.subTest(length=len(value)):
                self.assertTrue(
                    is_safe_public_metadata(
                        value,
                        max_length=MAX_REPORT_STRING_LENGTH,
                    )
                )
                self.assertIsNone(validate_report_structure({"value": value}))

        too_long = maximum_ascii + "a"
        self.assertFalse(
            is_safe_public_metadata(
                too_long,
                max_length=MAX_REPORT_STRING_LENGTH,
            )
        )
        self.assertEqual(
            validate_report_structure({"value": too_long}),
            "report_structure_unsafe_string",
        )

    def test_cycles_and_deep_inputs_never_escape_compare_or_golden(self) -> None:
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        cases = (
            ("cycle", cycle, "report_structure_cycle"),
            (
                "depth",
                _nested_list(MAX_REPORT_DEPTH + 1),
                "report_structure_too_deep",
            ),
            (
                "nodes",
                _node_list(MAX_REPORT_NODES + 1),
                "report_structure_too_large",
            ),
            ("non_json", object(), "report_structure_non_json_type"),
            ("nonfinite", math.nan, "report_structure_nonfinite_number"),
            (
                "oversize",
                MAX_SAFE_NUMERIC_MAGNITUDE + 1,
                "report_structure_oversize_number",
            ),
            ("unsafe_string", "private\ud800value", "report_structure_unsafe_string"),
        )
        for name, value, reason in cases:
            with self.subTest(path="compare", case=name):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                baseline["untrusted_extra"] = value
                result = compare_reports(baseline, candidate)
                self.assertEqual(
                    result["compatibility"]["reasons"], ["baseline_" + reason]
                )
                if isinstance(value, str):
                    _assert_not_reflected(self, value, result)

            with self.subTest(path="golden", case=name):
                reports = _golden_sequence()
                reports[0]["untrusted_extra"] = value
                result = validate_golden_sequence(reports)
                self.assertEqual(result["status"], "ineligible")
                self.assertIn("run_1_" + reason, result["eligibility_reasons"])
                self.assertIsNone(result["run_fingerprints"][0])
                self.assertTrue(
                    all(
                        fingerprint is not None
                        for fingerprint in result["run_fingerprints"][1:]
                    )
                )
                if isinstance(value, str):
                    _assert_not_reflected(self, value, result)

    def test_loaded_structure_failures_are_generic_and_nonreflective(self) -> None:
        secret = "PRIVATE_DEEP_JSON_PAYLOAD"
        raw = '{"nested":' * (MAX_REPORT_DEPTH + 1)
        raw += json.dumps(secret)
        raw += "}" * (MAX_REPORT_DEPTH + 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "too-deep.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ComparisonInputError) as raised:
                load_report(path)
            self.assertEqual(str(raised.exception), "report_structure_too_deep")
            _assert_not_reflected(self, secret, raised.exception)

    def test_loaded_unsafe_strings_and_numbers_return_only_fixed_codes(self) -> None:
        secret = "runtime token=PRIVATE_LOADED_TOKEN"
        cases = (
            (
                "unsafe-string",
                json.dumps({"value": secret}),
                "report_structure_unsafe_string",
            ),
            (
                "nan",
                '{"value":NaN}',
                "saved report is unreadable or not valid JSON",
            ),
            ("overflow-float", '{"value":1e999}', "report_structure_nonfinite_number"),
            (
                "oversize-int",
                '{"value":9007199254740992}',
                "report_structure_oversize_number",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, payload, reason in cases:
                with self.subTest(name=name):
                    path = Path(temp_dir) / f"{name}.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ComparisonInputError) as raised:
                        load_report(path)
                    self.assertEqual(str(raised.exception), reason)
                    if name == "unsafe-string":
                        _assert_not_reflected(self, secret, raised.exception)


if __name__ == "__main__":
    unittest.main()
