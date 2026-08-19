# Throttle five-stack live validation — 2026-08-19

Pod: `o4zw3lkzv9yr69` · GPU: NVIDIA GeForce RTX 4090 (24,564 MiB) ·
RunPod observed rate: $0.74/hour · Throttle: 0.2.0.

## Summary

| Project | Model used | Setup succeeded? | Throttle ran successfully? | Overall verdict | Sanity-check match? |
|---|---|---:|---:|---|---|
| vLLM 0.27.1 | Qwen/Qwen3-8B (already deployed; not one of the requested replacement choices) | Yes | Yes | Rejected / decision-ineligible | Yes |
| TGI 3.3.5 image | Qwen/Qwen2.5-3B-Instruct | No | No | Not run | N/A |
| SGLang `latest-runtime` | Qwen/Qwen2.5-3B-Instruct | Yes | Yes | Rejected / decision-ineligible | Yes |
| Ollama 0.32.14 | qwen2.5:3b | Yes | Yes | Rejected / decision-ineligible | Partial: c1 yes; c4 direction yes but magnitude drifted |
| LMDeploy `latest` | Qwen/Qwen2.5-3B-Instruct | Yes | Yes | Rejected / decision-ineligible | Yes |

All four completed Throttle reports have `status: complete`, both conditions
`decision_grade: true`, and the *overall* `decision_eligible: false`. Their common
overall rejection reasons are:

- `best_tested_result_is_not_statistically_supported`
- `complete_runtime_provenance_required`
- `immutable_image_digest_required`
- `immutable_model_revision_required`
- `runtime_verified_engine_flags_required`

Their `best_tested` state is `inconclusive`, with reasons
`multi_condition_order_not_counterbalanced` and `search_boundary_reached`.

## vLLM

The pre-existing endpoint served `Qwen/Qwen3-8B`, not Qwen2.5-3B or Mistral-7B.
It was confirmed rather than redeployed. The first requested-model curl was a real
404 and is preserved; the curl using the actually deployed model returned HTTP 200.
Throttle completed 201/201 valid measured requests at each concurrency. c1:
38.226 tok/s, p95 E2E 1279.648 ms, p95 TTFT 746.320 ms, $5.37745/1M output
tokens. c4: 143.919 tok/s, p95 E2E 1308.872 ms, p95 TTFT 746.468 ms,
$1.42853/1M output tokens. The separate 30-request checks measured 38.241 and
132.180 tok/s, with p95 E2E 1424.819 and 1480.945 ms. This is a reasonable sanity
match, including the throughput direction; it does not turn the inconclusive gate
into a decision.

## Text Generation Inference (TGI)

Setup failed. The pod was switched to the official
`ghcr.io/huggingface/text-generation-inference:3.3.5` image with
`Qwen/Qwen2.5-3B-Instruct`. The container exited, and a single explicit start attempt
exited again within one second. Exact observable state: `desiredStatus: EXITED`,
`runtimeStatus: stopped`, `runtimeStatusReason: stopped_by_runpod`; the required chat
curl returned bare HTTP 404. RunPod removed the container before SSH was available,
so no application stderr was retrievable. No Throttle run or manual load test was
attempted after the failed alive check.

## SGLang

The official `lmsysorg/sglang:latest-runtime` image launched
`Qwen/Qwen2.5-3B-Instruct`, and the raw chat curl returned exactly
`throttle-alive`. The first Throttle invocation failed pre-traffic because Throttle
requires a non-empty API-key environment variable even for an unauthenticated
endpoint; the second used an ignored dummy bearer and completed. c1: 66.630 tok/s,
p95 E2E 948.867 ms, p95 TTFT 755.131 ms, $3.08606/1M output tokens. c4:
230.490 tok/s, p95 E2E 806.228 ms, p95 TTFT 553.660 ms, $0.89193/1M output
tokens. Manual: 65.002 and 231.843 tok/s; p95 E2E 931.856 and 1042.896 ms.
Sanity check matched.

## Ollama

Ollama 0.32.14 launched on port 8000 and successfully pulled the public
`qwen2.5:3b` artifact (the complete pull transcript remains at
`/workspace/throttle-five-stack-20260819/04-ollama-pull-raw.log` because the pod's
direct SCP port refused connections). The raw chat curl returned exactly
`throttle-alive`. c1: 43.783 tok/s, p95 E2E 1211.116 ms, p95 TTFT 1074.128 ms,
$4.73399/1M output tokens. c4: 172.522 tok/s, p95 E2E 1090.804 ms, p95 TTFT
982.134 ms, $1.19189/1M output tokens. Manual: 43.604 and 152.528 tok/s; p95
E2E 1354.244 and 1358.729 ms. c1 aligned, but manual c4 throughput was 11.6%
lower and p95 latency about 24.6% higher than Throttle; direction held, magnitude
did not align tightly.

## LMDeploy

The official `openmmlab/lmdeploy:latest` image launched
`Qwen/Qwen2.5-3B-Instruct`; the raw chat curl returned exactly
`throttle-alive`. c1: 61.086 tok/s, p95 E2E 979.807 ms, p95 TTFT 729.384 ms,
$3.36502/1M output tokens. c4: 238.318 tok/s, p95 E2E 813.886 ms, p95 TTFT
557.117 ms, $0.86274/1M output tokens. Manual: 60.132 and 230.987 tok/s; p95
E2E 1051.289 and 781.306 ms. Sanity check matched.

## Accuracy read

Throttle gave no false overall decision-grade result: every complete multi-condition
report was correctly decision-ineligible and inconclusive. There is no demonstrated
false rejection either. The manual checks support the throughput direction on all
four completed stacks, but they cannot satisfy the missing immutable provenance,
counterbalancing, or boundary-expansion requirements. Ollama's c4 drift is precisely
why the rejection should not be overridden.

The terminal table's per-condition `grade yes` can be mistaken for an overall
decision-grade verdict even though the JSON says `decision_eligible: false` and the
terminal conclusion says inconclusive. This terminology is a presentation risk, not
a false result in the saved artifact.

## Trust blockers, ranked

1. **Overall verdict is too easy to misread.** The terminal prints `grade yes` for
   conditions but does not print the overall `decision_eligible: false` or its five
   exact reasons. A screenshot can overstate the result.
2. **A normal c1/c4 run can never support the apparent choice.** Condition-major,
   non-counterbalanced order plus a winning upper boundary forces inconclusive. The
   CLI should guide users directly toward a counterbalanced comparison or range
   expansion instead of merely printing a descriptive best value.
3. **Unauthenticated endpoints require a fake key value.** SGLang, Ollama, and
   LMDeploy accepted no-auth curl, but Throttle refused to start without a non-empty
   key environment variable. This is unnecessary friction and encourages dummy
   secrets in automation.
4. **Decision-grade provenance is operationally hard across stacks.** Immutable image
   digest, full model revision, runtime versions, GPU fingerprint, and runtime-verified
   engine flags were not discoverable from the endpoints. The strict rejection is
   correct, but the tool needs stack-specific evidence capture or clearer collection
   instructions to make cross-stack proof practical.
5. **Sanity-check variance needs an explicit tolerance policy.** Ollama c4 differed
   materially between the long run and 30-request check. Throttle did not falsely
   certify it, but users need a defined external-replication threshold rather than
   an informal “roughly lines up.”

## Cost-saving angle

There is no evidence-backed configuration win to claim because all four overall
results were rejected/inconclusive. Descriptively only, c4 reduced measured
cost-per-million-output-tokens versus c1 by 73.4% (vLLM), 71.1% (SGLang), 74.8%
(Ollama), and 74.4% (LMDeploy), at the same observed $0.74/hour pod rate. Those are
measured observations, not decision-grade savings claims. No throughput-per-dollar
claim should be published from these runs.

## Final verdict

Throttle is **not ready as proof that it works across real serving stacks**. It did
successfully drive and strictly validate 1,608 measured streaming responses across
vLLM, SGLang, Ollama, and LMDeploy, and its results mostly replicated in a separate
client. But it produced no overall decision-grade comparison, TGI setup failed, its
terminal wording is easy to overread, and no completed run had the immutable/runtime
evidence required by its own gate. It is ready to demonstrate narrow measurement and
response-validation compatibility on four stacks, not cross-stack optimization proof.

## Evidence handling

Raw files and four sanitized Throttle JSON reports are in this directory. Some
control-plane logs contain pod environment values and must not be shared publicly
without redaction. Zero-byte files are retained where the HTTP response had an empty
body (for example TGI's bare 404); the associated status is documented above and in
the control-plane state logs. No benchmark process remains running. The pod is still
running LMDeploy and continues billing until the operator stops it.
