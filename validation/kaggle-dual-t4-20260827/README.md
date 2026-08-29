> External validation performed against Throttle commit
> `7c470078de78f4771b190db366be391e9f8b6e44`.
>
> Hardware: Kaggle 2× Tesla T4, SM75, TP=2.
>
> The local URL modification used to continue `validate-sim`
> was an experimental diagnostic patch and is not the final
> implementation proposed by this PR.

# Throttle validation on Kaggle dual NVIDIA Tesla T4

## Environment

- Throttle: 0.3.0
- kaggle-vllm: 0.1.2
- Backend: upstream-derived vLLM runtime
- GPU: 2 x NVIDIA Tesla T4
- Compute capability: 7.5 / SM75
- Tensor parallelism: TP=2
- Model: Qwen2.5-3B sharded_state
- API: OpenAI-compatible vLLM endpoint
- Metrics: vLLM Prometheus /metrics

## Baseline

- vLLM server health: PASS
- /v1/models: PASS
- /v1/chat/completions: PASS
- /metrics: PASS
- TP workers: PASS
- Throttle smoke: PASS

## Throttle smoke

Smoke completed successfully for tested concurrency:
- 1
- 4
- 8

This is smoke evidence only and not a production recommendation.

## validate-sim — unmodified Throttle

Unmodified validate-sim failed during Light load with HTTP 404.

Root cause observed in Throttle 0.3.0:
inconsistent construction of /v1/chat/completions when --endpoint-url
already includes /v1.

## Local validation patch

A local test-only URL fix was applied so validate-sim could proceed.
See:

validate-sim-local-fix.diff

## validate-sim — patched local test

### Light load

- Simulated wall clock: 22.11 s
- Measured wall clock: 24.70 s
- Error: -10.5%

- Simulated input throughput: 204.4 tok/s
- Measured input throughput: 207.2 tok/s
- Error: -1.4%

- Simulated output throughput: 142.2 tok/s
- Measured output throughput: 50.3 tok/s
- Error: +182.8%

- Simulated input cost: $1.36/M input tokens
- Measured normalized input cost: $1.34/M
- Error: +1.4%

Peak measured concurrency: 8

### Medium load

- Simulated wall clock: 18.58 s
- Measured wall clock: 19.96 s
- Error: -6.9%

- Simulated input throughput: 504.7 tok/s
- Measured input throughput: 544.8 tok/s
- Error: -7.4%

- Simulated output throughput: 423.7 tok/s
- Measured output throughput: 154.3 tok/s
- Error: +174.5%

- Simulated input cost: $0.55/M input tokens
- Measured normalized input cost: $0.51/M
- Error: +7.9%

Peak measured concurrency: 50

### Heavy load

Heavy workload failed at request 86 with HTTP 400.

The failure was preserved rather than patched further.

## Additional issue observed

During validate-sim shutdown, concurrent async request tasks emitted:

ValueError:
second argument (exceptions) must be a non-empty sequence

This occurred inside the AnyIO/httpcore connection path during asyncio shutdown.

## Interpretation

On this Turing/SM75 TP=2 environment:

- simulator wall-clock estimates were within roughly 7-11%
- simulator input throughput estimates were within roughly 1-8%
- simulator output throughput was overpredicted by roughly 175-183%

This suggests the current cost/performance simulator does not transfer cleanly
from its existing high-end GPU validation to dual Tesla T4 output-generation
performance.

The $1.00 GPU-hour value used by validate-sim was a synthetic normalization
input required by the CLI and does not represent Kaggle pricing.
