# Critique of docs/RESULT_STORE_DESIGN.md

Checked against the real artifacts in `validation/golden-live-20260817/`
(`golden.json`, `B1.json`, `RUN_AUDIT.md`). Findings below; the alternative
schema this motivated is in `RESULT_STORE_DESIGN_PROPOSAL.md`.

## Provenance doesn't exist anywhere in the JSON artifacts

The requirement is explicit: who ran it, what environment, owned or rented
hardware, on every record. No field in `golden.json` or `B1.json`'s `manifest`
captures any of this. It exists today only as prose in `RUN_AUDIT.md`
("the operator verified...", "RunPod pod... $1.39/hour"), written by hand,
per run, not structured or queryable. The original design's schema has no
field for operator identity and no owned/rented flag. This is a hard blocker,
not a nice-to-have.

## Model revision is dropped by the design's own normalization rule

`docs/RESULT_STORE_DESIGN.md`'s config-key rule: *"Model: Extract family name
(Qwen2.5, Llama-3.1, etc.)"*. But `manifest.model` in `B1.json` is:

```json
{"id": "Qwen/Qwen2.5-0.5B-Instruct", "immutable_revision": "7ae557604adf67be50417f59c2c2f167def9a775"}
```

The revision is already there, already used in `RUN_AUDIT.md`
("model revision: `7ae557604adf...`"). The stated requirement is "model
**and revision**" as a key dimension. The design's own stated rule deletes it.

## `backend` names two different things

The design's config key includes `backend`, with examples `vllm`/`ollama`/
`sglang`, the serving engine. The real manifest already has a field with that
exact name, `manifest.engine.backend`, and its value in both `B1.json` and
`C1.json` is `"native"`, Throttle's own request-execution client (native
httpx vs. guidellm), not the serving engine. The serving engine version is a
different field: `manifest.engine.server_version` ("0.16.0"). Implemented
literally, every run would key on `backend=native` regardless of the actual
serving engine, which is exactly the "silently treat a different backend as
the same config" failure the requirements explicitly forbid.

## GPU normalization drops distinctions already present in the data

Stated rule: *"GPU: Normalize vendor + tier + VRAM (A100-80GB,
RTX-4090-24GB)"*. Real data has `manifest.runtime.gpu = "NVIDIA A100 80GB
PCIe"` and `manifest.cost.gpu_count = 1`, both present, both dropped by the
stated normalization. PCIe vs. SXM and single vs. multi-GPU materially change
performance; collapsing them together is the same class of silent-conflation
risk called out for backend versions.

## "Near match must name what differs" isn't buildable from what the design stores

The index only stores an opaque `config_key` (for workload specifically, a
truncated SHA256). Two hashes that don't match can't be diffed to say what
changed. The real manifest data is rich and diffable field by field
(`model.immutable_revision`, `runtime.gpu`, `engine.server_version`,
`engine.effective_flags`, `workload.measured_sha256`), but the design
collapses all of it into one opaque key before any comparison happens, and
doesn't retain the underlying fields to diff later.

## No answer for multiple operators, which is the actual next use case

A local `~/.throttle/results/index.json`, plus a manual `export-results`/
`import-results --merge` command listed under future work, doesn't give
"check for a prior result before spending GPU time" across operators who
haven't manually exchanged files first. Concurrent writes to one shared index
file also aren't addressed. Given results are about to be produced on other
people's machines, this is the actual use case, not a future enhancement.

## Minor: the storage function's inputs don't match where the data actually lives

`store_result(result_file, config, metadata)` implies the metadata is
naturally available alongside a single result file. For golden runs it isn't:
`golden.json` (the summary) carries none of model/GPU/engine info; that only
exists in the six position files (`B1.json`...`C3.json`). Any implementation
needs to know to reach into a different file than the one being stored.

## What holds up

- JSON-over-SQLite is a reasonable call given the "portable, inspectable"
  requirement, no objection to the format itself.
- The workload-hash idea maps cleanly to real fields
  (`measured_sha256`/`warmup_sha256`); that part of the design is realistic.
