from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from throttle.result_store import (
    MatchResult,
    Provenance,
    ResultStoreError,
    append_record,
    build_record,
    find_match,
    format_match_message,
    identity_matches,
    load_records,
    results_dirs,
)


def _identity(**overrides: object) -> dict[str, object]:
    base = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_count": 1,
        "engine_name": "vllm",
        "engine_version": "0.16.0",
    }
    base.update(overrides)
    return base


def _comparison(**overrides: object) -> dict[str, object]:
    base = {
        "changed_flag": "max_num_seqs",
        "baseline_value": "1",
        "candidate_value": "8",
        "measured_sha256": "787fc93d",
        "warmup_sha256": "b0c6f807",
        "cache_policy": "disabled",
    }
    base.update(overrides)
    return base


def _sample_record(**identity_overrides: object) -> dict[str, object]:
    return build_record(
        decision_eligible=True,
        decision_state="supported",
        overall_outcome="candidate_higher_throughput",
        throughput_delta_percent_estimate=217.85,
        throughput_delta_percent_low=189.47,
        throughput_delta_percent_high=246.22,
        **_identity(**identity_overrides),
        throttle_client_backend="native",
        throttle_client_backend_version="native-protocol-1",
        cuda_version="12.8",
        driver_version="550.127.05",
        image_digest="runpod/pytorch@sha256:abc",
        **_comparison(),
        measured_prompt_count=8,
        warmup_prompt_count=3,
        seed=42,
        source_run_fingerprints=["a", "b"],
        artifact_paths=["golden.json"],
        cost_usd_estimate=0.74,
        result_id="deadbeef",
        provenance=Provenance(operator="dhruv", hardware_ownership="rented"),
        gpu_fingerprint_sha256="8d76b604",
    )


class ProvenanceTests(unittest.TestCase):
    def test_missing_operator_is_rejected(self) -> None:
        with self.assertRaises(ResultStoreError):
            Provenance(operator="", hardware_ownership="owned")

    def test_invalid_hardware_ownership_is_rejected(self) -> None:
        with self.assertRaises(ResultStoreError):
            Provenance(operator="dhruv", hardware_ownership="borrowed")

    def test_valid_provenance_is_accepted(self) -> None:
        prov = Provenance(operator="dhruv", hardware_ownership="owned")
        self.assertEqual(prov.operator, "dhruv")
        self.assertFalse(prov.backfilled)


class BuildRecordTests(unittest.TestCase):
    def test_record_shape_matches_schema(self) -> None:
        record = _sample_record()
        self.assertEqual(record["record_version"], 1)
        self.assertEqual(record["identity"]["model_id"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(record["parameter_change"]["changed_flag"], "max_num_seqs")
        self.assertEqual(record["provenance"]["operator"], "dhruv")
        self.assertEqual(record["provenance"]["hardware_ownership"], "rented")
        self.assertFalse(record["provenance"]["backfilled"])
        self.assertEqual(record["workload"]["seed"], 42)


class StorageRoundTripTests(unittest.TestCase):
    def test_append_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            record = _sample_record()
            path = append_record(record, results_dir=results_dir)
            self.assertTrue(path.exists())
            loaded = load_records([results_dir])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["result_id"], record["result_id"])

    def test_append_never_truncates_existing_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            first = _sample_record()
            second = _sample_record()
            second["result_id"] = "second"
            append_record(first, results_dir=results_dir)
            append_record(second, results_dir=results_dir)
            loaded = load_records([results_dir])
            self.assertEqual(len(loaded), 2)
            ids = {r["result_id"] for r in loaded}
            self.assertEqual(ids, {"deadbeef", "second"})

    def test_corrupt_line_is_skipped_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            record = _sample_record()
            path = append_record(record, results_dir=results_dir)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("{not valid json\n")
            loaded = load_records([results_dir])
            self.assertEqual(len(loaded), 1)

    def test_results_dirs_reads_extra_env_var(self) -> None:
        with mock.patch.dict(
            "os.environ", {"THROTTLE_RESULTS_DIRS": "/tmp/shared-a:/tmp/shared-b"}
        ):
            dirs = results_dirs()
            self.assertIn(Path("/tmp/shared-a"), dirs)
            self.assertIn(Path("/tmp/shared-b"), dirs)


class IdentityMatchTests(unittest.TestCase):
    def test_identical_identity_matches(self) -> None:
        a = {"identity": _identity()}
        b = {"identity": _identity()}
        self.assertTrue(identity_matches(a, b))

    def test_different_gpu_does_not_match(self) -> None:
        a = {"identity": _identity()}
        b = {"identity": _identity(gpu="NVIDIA A100 80GB SXM4")}
        self.assertFalse(identity_matches(a, b))

    def test_different_engine_version_does_not_match(self) -> None:
        a = {"identity": _identity()}
        b = {"identity": _identity(engine_version="0.17.0")}
        self.assertFalse(identity_matches(a, b))

    def test_unknown_on_either_side_fails_closed(self) -> None:
        a = {"identity": _identity()}
        b = {"identity": _identity(engine_name="unknown")}
        self.assertFalse(identity_matches(a, b))
        c = {"identity": _identity(gpu="unknown")}
        self.assertFalse(identity_matches(a, c))

    def test_missing_field_fails_closed(self) -> None:
        # Simulates an older-schema record that predates a field entirely,
        # not just one set to "unknown".
        a = {"identity": _identity()}
        older = _identity()
        del older["engine_name"]
        b = {"identity": older}
        self.assertFalse(identity_matches(a, b))


class FindMatchTests(unittest.TestCase):
    def test_no_prior_records_returns_none(self) -> None:
        result = find_match(_identity(), _comparison(), [])
        self.assertIsNone(result)

    def test_no_identity_overlap_returns_none(self) -> None:
        prior = _sample_record(gpu="NVIDIA H100 80GB")
        result = find_match(_identity(), _comparison(), [prior])
        self.assertIsNone(result)

    def test_exact_match(self) -> None:
        prior = _sample_record()
        result = find_match(_identity(), _comparison(), [prior])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.exact)
        self.assertEqual(result.differing_fields, ())

    def test_near_match_names_the_differing_field(self) -> None:
        prior = _sample_record()
        candidate_comparison = _comparison(cache_policy="warm")
        result = find_match(_identity(), candidate_comparison, [prior])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.exact)
        self.assertEqual(result.differing_fields, ("cache_policy",))

    def test_near_match_can_name_multiple_differing_fields(self) -> None:
        prior = _sample_record()
        candidate_comparison = _comparison(cache_policy="warm", warmup_sha256="different")
        result = find_match(_identity(), candidate_comparison, [prior])
        assert result is not None
        self.assertEqual(
            set(result.differing_fields), {"cache_policy", "warmup_sha256"}
        )

    def test_picks_most_recent_prior_on_multiple_identity_matches(self) -> None:
        older = _sample_record()
        older["created_at"] = "2026-01-01T00:00:00+00:00"
        older["result_id"] = "older"
        newer = _sample_record()
        newer["created_at"] = "2026-06-01T00:00:00+00:00"
        newer["result_id"] = "newer"
        result = find_match(_identity(), _comparison(), [older, newer])
        assert result is not None
        self.assertEqual(result.prior["result_id"], "newer")


class FormatMatchMessageTests(unittest.TestCase):
    def test_exact_match_message_names_the_outcome(self) -> None:
        prior = _sample_record()
        result = MatchResult(prior=prior, exact=True, differing_fields=())
        message = format_match_message(result)
        self.assertIn("EXACT MATCH", message)
        self.assertIn("candidate_higher_throughput", message)

    def test_near_match_message_names_each_differing_field(self) -> None:
        prior = _sample_record()
        result = MatchResult(
            prior=prior, exact=False, differing_fields=("cache_policy", "warmup_sha256")
        )
        message = format_match_message(result)
        self.assertIn("NEAR MATCH", message)
        self.assertIn("cache_policy", message)
        self.assertIn("warmup_sha256", message)


if __name__ == "__main__":
    unittest.main()
