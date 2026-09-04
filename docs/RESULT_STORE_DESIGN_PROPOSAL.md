# Result Store Design - Proposal (supersedes docs/RESULT_STORE_DESIGN.md)

**Status**: implemented in this PR. Written after reviewing `docs/RESULT_STORE_DESIGN.md`
against the real artifacts in `validation/golden-live-20260817/`. That review is in
`RESULT_STORE_DESIGN_CRITIQUE.md`; the short version: the original design's schema
doesn't match what Throttle already produces, drops fields it's explicitly required to
keep (model revision), reuses a field name (`backend`) that already means something
else in the real manifest, and has no answer for multiple operators contributing
results. This doc starts from the real `manifest` shape already sitting in every
artifact today, since it's more complete than anything the original design invented.

---

## Design principle

Don't generate an opaque key and discard the fields that produced it. Store the
fields themselves. Matching and "what differs" both fall out of comparing real
fields; neither is possible once everything's been collapsed into one hash.

---

## Record shape

One JSON object per decision-eligible run (source: the run's `golden.json`/
`benchmark.json` plus the per-position manifest, e.g. `B1.json`, which already
carries almost everything below):

```json
{
  "record_version": 1,
  "result_id": "sha256 of the primary artifact file",
  "artifact_type": "throttle_golden_live_comparison",
  "decision_eligible": true,
  "decision_state": "supported",
  "created_at": "2026-08-17T05:00:07Z",
  "provenance": {
    "operator": "kush@runpod-a100-01",
    "environment_note": "RunPod pod, on-demand, deleted after run",
    "hardware_ownership": "rented",
    "hardware_provider": "runpod",
    "hardware_rate_usd_per_hour": 1.39
  },
  "identity": {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "gpu": "NVIDIA A100 80GB PCIe",
    "gpu_count": 1,
    "gpu_fingerprint_sha256": "8d76b6046b6e83c5a41f9d565f47c2b64c209bcfd3a5b07a663e5809646ccb0a",
    "engine_name": "vllm",
    "engine_version": "0.16.0",
    "throttle_client_backend": "native",
    "cuda_version": "12.8",
    "driver_version": "550.127.05",
    "image_digest": "runpod/pytorch@sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
  },
  "parameter_change": {
    "changed_flag": "max_num_seqs",
    "baseline_value": "1",
    "candidate_value": "8"
  },
  "workload": {
    "type": "standard",
    "measured_sha256": "787fc93d62e140eba0ae9d2baaf16c7d386c536d05f525f1c01b3d955bc70871",
    "warmup_sha256": "b0c6f807ea9f25517ffd7b11d3fea29adfd25c8bd3561f58f32ecc7e1b1cb352",
    "measured_prompt_count": 8,
    "warmup_prompt_count": 3
  },
  "outcome": {
    "overall_outcome": "candidate_higher_throughput",
    "throughput_delta_percent_estimate": 217.85,
    "throughput_delta_percent_low": 189.47,
    "throughput_delta_percent_high": 246.22
  },
  "artifact_paths": ["golden.json", "B1.json", "C1.json", "B2.json", "C2.json", "B3.json", "C3.json"],
  "cost_usd_estimate": 0.74
}
```

Every field above except `provenance.*` and `result_id`/`created_at`/`artifact_paths`
already exists today in `manifest` (see `B1.json`) or the summary artifact
(`golden.json`). `provenance` is the only genuinely new data this design asks
operators to supply.

### What changed from the original design, and why

- **`identity.model_revision` is present and required.** The original design's
  normalization rule discarded it ("extract family name"). It's already sitting in
  `manifest.model.immutable_revision`. Keeping it is free.
- **`identity.gpu` keeps the full string, plus `identity.gpu_count`.** The original
  design's rule was "vendor + tier + VRAM," which would treat PCIe and SXM A100s, or
  a 1x and 8x setup, as the same GPU. Both distinctions are already in
  `manifest.runtime.gpu` and `manifest.cost.gpu_count`.
- **`engine_name`/`engine_version` replace the overloaded `backend` field.** The real
  manifest already has a field called `manifest.engine.backend`, and its value is
  `"native"` for both baseline and candidate positions I checked, that's Throttle's
  own request-execution client (native vs. guidellm), not the model server. The
  model server version is `manifest.engine.server_version`. The original design's
  config key assumed `backend` meant `vllm`/`ollama`/`sglang`. Wired up literally, it
  would key every run on `"native"` and never actually distinguish serving engines.
  This proposal keeps `throttle_client_backend` (native/guidellm) as its own field,
  separate from `engine_name`/`engine_version` (the thing actually being benchmarked).
  As of this PR, engine_name/engine_version are now a real structured field in the
  manifest (see PR #29), closing the gap this critique identified.
- **No generated `config_key` string.** Matching and diffing both operate on the
  `identity`/`parameter_change`/`workload` objects directly (see below), not on a
  precomputed hash.
- **`provenance` is a new, mandatory block.** Nothing in the current artifacts
  captures who ran a result or whether the hardware was owned or rented; today that
  only exists as hand-written prose in a `RUN_AUDIT.md` per run. This makes it
  structured and required.

---

## Matching and near-matching

Two fixed field sets:

- **Identity fields** (`identity.model_id`, `identity.model_revision`,
  `identity.gpu`, `identity.gpu_count`, `identity.engine_name`,
  `identity.engine_version`): if *any* of these differ from a candidate run, the
  prior result is not offered as reusable, full stop. This is the "never silently
  treat a different GPU or backend version as the same config" rule, enforced by
  comparing real values instead of a hash.
- **Comparison fields** (`parameter_change`, `workload`): if identity fields all
  match but one or more comparison fields differ, that's a **near match**. Print
  the prior result *and* the specific field(s) that differ, by name, with old and
  new values. Example:

  ```
  ⚠ NEAR MATCH — same model/GPU/engine, different workload
    Prior workload.measured_sha256: 787fc93d...
    This run's workload.measured_sha256: a91cd002...
    (parameter_change is identical: max_num_seqs 1 → 8)
  Previous run: 2026-08-17 05:00 UTC
    Outcome: +217.8% throughput (CI: +189.5% to +246.2%)
  [1] Use prior result anyway (workload differs, may not transfer)
  [2] Re-run
  [3] Cancel
  ```

- **Exact match** (identity and comparison fields all equal): offer to skip the run
  entirely, same UX as the original design proposed.

This requires storing the actual field values, not a hash, which the record shape
above already does.

---

## Provenance: what's required, and how it's supplied without being annoying

`hardware_ownership` (`"owned"` or `"rented"`) and `operator` are required on every
stored record; a run that can't determine them isn't stored, matching the project's
existing fail-closed conventions rather than silently guessing.

To avoid asking on every single run:

- `operator` defaults to `$USER@$(hostname)`, overridable via `--operator` or an
  `THROTTLE_OPERATOR` env var.
- `hardware_ownership` is set once per machine, in `~/.throttle/config.yaml`
  (`hardware-ownership: owned` or `rented`), overridable per run with
  `--owned`/`--rented`. A machine typically doesn't change category run to run, so
  this is a one-time setup step, not a per-run prompt.
- `hardware_provider` and `hardware_rate_usd_per_hour` are optional and only make
  sense when rented; already-computed cost fields (`manifest.cost.total_hourly_rate`)
  can populate the rate automatically when present.

---

## Storage: append-only files, not one shared mutable index

The original design's single `~/.throttle/results/index.json`, read-modified-written
on every run, has two problems once more than one operator is involved (which is the
explicit next step here): concurrent writes to one file race, and there's no answer
for reconciling two machines' histories beyond a manual `export`/`import --merge`
command.

Implemented: each record is appended as one line to an NDJSON file, sharded by month to
keep individual files small and diffable:

```
~/.throttle/results/2026-08.ndjson
~/.throttle/results/2026-09.ndjson
```

- **Local-only by default.** Nothing changes for a solo user; results just
  accumulate locally.
- **Sharing is "point at more files," not a custom protocol.** A
  `THROTTLE_RESULTS_DIRS` env var (or `--results-dir`, repeatable) tells Throttle to
  also read NDJSON files from other paths, e.g. a shared network drive or a checked
  out git repo the team pushes results to. Reconciling two machines' histories is
  then just `git pull`, appends never conflict the way a mutable index would.
- **Still portable and inspectable**: plain text, one JSON object per line, greppable
  and `jq`-able, no database dependency, matches what was asked for.

This is a bigger change from the original design than the schema itself, but it's
the part that actually has to work for "running this on other people's machines" to
mean anything.

---

## Open questions for discussion

1. Is NDJSON + directory-of-sources the right shared-store answer, or is a
   lightweight SQLite file (still a single portable file, still inspectable with the
   `sqlite3` CLI) preferable despite the write-concurrency question? Worth arguing
   through before committing either way.
2. Should non-decision-eligible runs be recorded at all (the original design's
   `decision_eligible: false` entries), and if so, do they need full provenance too,
   or a lighter record?
3. `parameter_change.changed_flag` only handles a single-flag comparison
   (`optimization_credit.changed_flag` in the real artifacts is likewise
   singular). Golden protocol appears to be single-flag by design today, but worth
   confirming this doesn't need to support multi-flag changes before locking the
   schema.
