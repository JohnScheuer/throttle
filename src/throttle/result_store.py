"""Persistent result store for Throttle golden runs.

Every decision-eligible golden comparison is appended as one JSON record to
an append-only NDJSON file, sharded by month. Records are never rewritten in
place: a schema change adds new fields as optional, and a record that
predates a field carries an explicit unknown value for it rather than a
silent default. Matching logic treats unknown as "can't safely claim a
match" and fails closed.

Design rationale and the full schema discussion live in
docs/RESULT_STORE_DESIGN_PROPOSAL.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

RECORD_VERSION = 1

DEFAULT_RESULTS_DIR = Path.home() / ".throttle" / "results"

# If any of these differ between a candidate run and a prior record, the
# prior record is never offered as reusable, full stop. This is the "never
# silently treat a different GPU or backend version as the same config"
# rule: it's enforced by comparing the actual recorded values, not a
# generated key.
IDENTITY_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_revision",
    "gpu",
    "gpu_count",
    "engine_name",
    "engine_version",
)

# If every identity field matches but any of these differ, it's a near
# match: the prior record is still printed, but with the differing field(s)
# named explicitly rather than a bare "no match".
COMPARISON_FIELDS: tuple[str, ...] = (
    "changed_flag",
    "baseline_value",
    "candidate_value",
    "measured_sha256",
    "warmup_sha256",
    "cache_policy",
)


class ResultStoreError(ValueError):
    """Raised when a record can't be honestly built or stored."""


@dataclass(frozen=True)
class Provenance:
    """Provenance is mandatory on every record. There is deliberately no
    default for operator or hardware_ownership: a run whose provenance
    can't be determined is not stored, rather than stored with a guess.
    """

    operator: str
    hardware_ownership: str  # "owned" or "rented"
    environment_note: str = "unknown"
    hardware_provider: str = "unknown"
    hardware_rate_usd_per_hour: float | None = None
    backfilled: bool = False

    def __post_init__(self) -> None:
        if not self.operator or not self.operator.strip():
            raise ResultStoreError(
                "provenance.operator is required; a run whose operator can't "
                "be determined is not stored"
            )
        if self.hardware_ownership not in ("owned", "rented"):
            raise ResultStoreError(
                "provenance.hardware_ownership must be 'owned' or 'rented', "
                f"got {self.hardware_ownership!r}"
            )


def results_dirs() -> list[Path]:
    """Every directory to read and search records from: the local default,
    plus any extra directories named in THROTTLE_RESULTS_DIRS (colon
    separated), e.g. a synced team directory or a checked-out git repo of
    records. Sharing results across operators is "point at more paths," not
    a database or a custom sync protocol.
    """
    dirs = [DEFAULT_RESULTS_DIR]
    extra = os.environ.get("THROTTLE_RESULTS_DIRS", "")
    for raw in extra.split(":"):
        raw = raw.strip()
        if raw:
            dirs.append(Path(raw).expanduser())
    return dirs


def _shard_path(results_dir: Path, when: datetime) -> Path:
    return results_dir / f"{when.strftime('%Y-%m')}.ndjson"


def build_record(
    *,
    decision_eligible: bool,
    decision_state: str,
    overall_outcome: str,
    throughput_delta_percent_estimate: float | None,
    throughput_delta_percent_low: float | None,
    throughput_delta_percent_high: float | None,
    model_id: str,
    model_revision: str,
    gpu: str,
    gpu_count: int,
    gpu_fingerprint_sha256: str,
    engine_name: str,
    engine_version: str,
    throttle_client_backend: str,
    throttle_client_backend_version: str,
    cuda_version: str,
    driver_version: str,
    image_digest: str,
    changed_flag: str,
    baseline_value: str,
    candidate_value: str,
    measured_sha256: str,
    warmup_sha256: str,
    measured_prompt_count: int | None,
    warmup_prompt_count: int | None,
    seed: int | None,
    cache_policy: str,
    source_run_fingerprints: Sequence[str],
    artifact_paths: Sequence[str],
    cost_usd_estimate: float,
    result_id: str,
    provenance: Provenance,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one result-store record. Every field is passed explicitly
    rather than re-derived from an artifact here, so the caller (which
    already has the real manifest in hand) is the single source of truth
    for what actually ran; this function only assembles and validates the
    record shape.
    """
    return {
        "record_version": RECORD_VERSION,
        "result_id": result_id,
        "artifact_type": "throttle_golden_live_comparison",
        "decision_eligible": bool(decision_eligible),
        "decision_state": decision_state,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "operator": provenance.operator,
            "backfilled": provenance.backfilled,
            "environment_note": provenance.environment_note,
            "hardware_ownership": provenance.hardware_ownership,
            "hardware_provider": provenance.hardware_provider,
            "hardware_rate_usd_per_hour": provenance.hardware_rate_usd_per_hour,
        },
        "identity": {
            "model_id": model_id,
            "model_revision": model_revision,
            "gpu": gpu,
            "gpu_count": gpu_count,
            "gpu_fingerprint_sha256": gpu_fingerprint_sha256,
            "engine_name": engine_name,
            "engine_version": engine_version,
            "throttle_client_backend": throttle_client_backend,
            "throttle_client_backend_version": throttle_client_backend_version,
            "cuda_version": cuda_version,
            "driver_version": driver_version,
            "image_digest": image_digest,
        },
        "parameter_change": {
            "changed_flag": changed_flag,
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
        },
        "workload": {
            "measured_sha256": measured_sha256,
            "warmup_sha256": warmup_sha256,
            "measured_prompt_count": measured_prompt_count,
            "warmup_prompt_count": warmup_prompt_count,
            "seed": seed,
            "cache_policy": cache_policy,
        },
        "outcome": {
            "overall_outcome": overall_outcome,
            "throughput_delta_percent_estimate": throughput_delta_percent_estimate,
            "throughput_delta_percent_low": throughput_delta_percent_low,
            "throughput_delta_percent_high": throughput_delta_percent_high,
        },
        "source_run_fingerprints": list(source_run_fingerprints),
        "artifact_paths": list(artifact_paths),
        "cost_usd_estimate": cost_usd_estimate,
    }


def append_record(
    record: Mapping[str, Any], *, results_dir: Path | None = None
) -> Path:
    """Append one record as a single NDJSON line. Never rewrites or
    truncates an existing file; a write failure partway through never
    corrupts prior records since each is one atomic line append.
    """
    target_dir = results_dir or DEFAULT_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    created_at = record.get("created_at", "")
    try:
        when = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        when = datetime.now(timezone.utc)
    path = _shard_path(target_dir, when)
    line = json.dumps(record, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path


def load_records(dirs: Sequence[Path] | None = None) -> list[dict[str, Any]]:
    """Load every record from every configured results directory. A line
    that isn't valid JSON is skipped rather than aborting the whole load;
    corrupt or partial lines from an interrupted write shouldn't hide every
    other record in the file.
    """
    records: list[dict[str, Any]] = []
    for directory in dirs if dirs is not None else results_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.ndjson")):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def _identity_value(record: Mapping[str, Any], field: str) -> Any:
    return record.get("identity", {}).get(field)


def _comparison_value(record: Mapping[str, Any], field: str) -> Any:
    if field in ("changed_flag", "baseline_value", "candidate_value"):
        return record.get("parameter_change", {}).get(field)
    return record.get("workload", {}).get(field)


def identity_matches(
    candidate: Mapping[str, Any], prior: Mapping[str, Any]
) -> bool:
    """True only if every identity field is present, not 'unknown', and
    equal on both sides. An 'unknown' or missing value on either side means
    "can't safely claim a match" and fails closed rather than matching.
    """
    for field in IDENTITY_FIELDS:
        left = _identity_value(candidate, field)
        right = _identity_value(prior, field)
        if left is None or right is None:
            return False
        if left in ("unknown",) or right in ("unknown",):
            return False
        if left != right:
            return False
    return True


@dataclass(frozen=True)
class MatchResult:
    prior: dict[str, Any]
    exact: bool
    differing_fields: tuple[str, ...]


def find_match(
    candidate_identity: Mapping[str, Any],
    candidate_comparison: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> MatchResult | None:
    """Look for a prior record with the same identity as the candidate run.
    Returns the most recent identity match, exact if every comparison field
    also matches, otherwise near, with the differing fields named. Returns
    None if no prior record shares this run's identity at all, in which
    case there's nothing to reuse or warn about.
    """
    identity_probe = {"identity": dict(candidate_identity)}
    matches = [r for r in records if identity_matches(identity_probe, r)]
    if not matches:
        return None
    matches.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    prior = matches[0]
    differing: list[str] = []
    for field in COMPARISON_FIELDS:
        candidate_value = candidate_comparison.get(field)
        prior_value = _comparison_value(prior, field)
        if candidate_value != prior_value:
            differing.append(field)
    return MatchResult(prior=prior, exact=not differing, differing_fields=tuple(differing))


def format_match_message(match: MatchResult) -> str:
    """A human-readable line naming exactly what differs (or that nothing
    does), for printing before a golden run starts. Never a bare
    match/no-match: the whole value of the store is being able to say what
    changed, not just whether something did.
    """
    prior = match.prior
    outcome = prior.get("outcome", {})
    when = prior.get("created_at", "unknown time")
    if match.exact:
        return (
            f"EXACT MATCH \u2014 a prior result for this exact configuration "
            f"already exists ({when}): {outcome.get('overall_outcome', 'unknown')}, "
            f"{outcome.get('throughput_delta_percent_estimate')}% "
            f"(CI: {outcome.get('throughput_delta_percent_low')}% to "
            f"{outcome.get('throughput_delta_percent_high')}%)."
        )
    lines = [
        f"NEAR MATCH \u2014 same model/GPU/engine as a prior result ({when}), "
        f"but {len(match.differing_fields)} field(s) differ:"
    ]
    for field in match.differing_fields:
        prior_value = _comparison_value(prior, field)
        lines.append(f"  {field}: prior={prior_value!r}")
    lines.append(
        f"  Prior outcome: {outcome.get('overall_outcome', 'unknown')}, "
        f"{outcome.get('throughput_delta_percent_estimate')}%"
    )
    return "\n".join(lines)
