# Experimental tuning: offline vLLM compatibility proof

This directory is a deterministic, offline compatibility/orchestration proof
for Throttle's explicitly opt-in experimental tuning path. It is not a
live deployment result, a performance benchmark, a configuration
recommendation, a savings result, or proof of scheduler saturation.

The Prometheus family names and current v1 `engine`/`model_name` label contract
were checked against the official Apache-2.0-licensed vLLM v0.27.1 release at commit
[`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`](https://github.com/vllm-project/vllm/tree/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac):

- [vLLM metric definitions (authoritative for current labels)](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/metrics/loggers.py)
- [metrics design and simplified example exposition](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/docs/design/metrics.md)
- [official `/metrics` endpoint usage](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/docs/usage/metrics.md)
- [upstream real-server request-to-metrics and gauge/reset test methodology](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/tests/entrypoints/serve/instrumentator/test_metrics.py)

No vLLM source code or captured operator metrics are copied here. The five
exposition snapshots and all numeric values were independently authored for
Throttle's tests. They describe one synthetic eight-second window with five
observations, 40 finished requests, 400 generated tokens, 200 prompt tokens,
sampled running load of eight, sampled waiting load of four, 70% sampled
KV-cache use, and no preemptions. The exercised v0.27.1 family subset is:
request outcomes, generation and prompt counters, preemptions, running and
waiting gauges, KV-cache use, TTFT, request TPOT, token ITL, E2E, queue,
prefill, and decode histograms. The current primary fixture intentionally omits
the legacy `num_requests_swapped` family. Its purpose is to exercise the real
bounded collector, derivation, suggestion-only analyzer, independent safety
boundary, report-to-safety envelope pairing, and CLI serialization with no
network access.

The connected test uses one `httpx.MockTransport` for both 40 in-process
`/v1/chat/completions` fixture calls (three warm-ups plus 37 measured) and five
in-process `/metrics` scrapes. Chat completions mutate the exported counters, so the
collector delta must reconcile with the ordinary smoke report. The checked-in
snapshots separately retain simple parser/derivation evidence. TPOT, ITL,
prompt, prefill, and decode diagnostics remain supplementary and are not
inputs to the analyzer's candidate or any decision gate.

The CLI's `--attest-same-deployment-exclusive-metrics` flag means the operator
attests that the exporter belongs to the same inference deployment receiving
the fixture traffic and that no unrelated inference traffic reaches that
deployment during the sampled window. Without that explicit attestation, the
same metrics remain insufficient evidence and produce no suggestion.

The expected policy result is only an exploratory candidate test value:
`max_num_seqs=8` to `max_num_seqs=10` at declared closed-loop concurrency 16.
It must retain `decision_eligible: false`, `auto_apply: false`,
`golden_validation_performed: false`, and `golden_protocol_eligible: false`.
No configuration is applied, and a separate counterbalanced Golden run would
still be required before any decision.

[`expected-invariants.json`](expected-invariants.json) is the checked expected
safety-projection subset. The offline end-to-end test extracts those exact
fields from the projection nested in the experimental envelope and compares
them field-for-field. It independently recomputes the envelope's canonical
SHA-256 binding to the sanitized ordinary smoke report, rejects a one-byte
report mutation and cross-paired valid safety output, and checks that labels,
endpoints, credentials, and unrelated exporter fields are absent. The digest
is a content binding, not a signature or proof of a live run.

[`UPSTREAM.json`](UPSTREAM.json) records the pinned upstream paths and audited
source hashes. [`checksums.sha256`](checksums.sha256) covers this README, the
provenance record, expected invariants, and all five exposition snapshots.

## Reproduce

From the repository root, run the warning-strict offline test file:

```sh
PYTHONPATH=src python -W error -m unittest tests.test_experimental_tuning -v
```

The frozen fixture is expected to report `Ran 21 tests` and `OK`. This remains
synthetic, in-process verification and does not require or represent a live
GPU or vLLM deployment.
