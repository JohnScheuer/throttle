# Throttle Quickstart

Get from nothing to seeing GPU cost estimates in under 5 minutes.

## Prerequisites

- Python 3.11 or later
- No GPU required for the simulator demo

## Install

```bash
# Create a fresh virtual environment
python3 -m venv throttle-demo
source throttle-demo/bin/activate

# Install throttle
pip install throttle
```

## Run the Demo

The fastest way to see what throttle does is to run the simulator demo:

```bash
throttle demo
```

This command:
- Generates a sample workload (100 requests)
- Simulates vLLM-style GPU inference
- Compares self-hosted GPU costs to API pricing
- Completes in under 1 second
- Requires no GPU or network connection

### Example Output

```
Throttle GPU Cost Simulator Demo
============================================================

[SIMULATED] Generating sample workload...
[SIMULATED] Generated 100 requests

[SIMULATED] Simulator configuration:
[SIMULATED]   Model: 7B parameter (ASSUMED)
[SIMULATED]   GPU: A100 40GB @ $1.50/hour
[SIMULATED]   Prefill throughput: 5000 tok/sec (ASSUMED)
[SIMULATED]   Decode throughput: 100 tok/sec (ASSUMED)
[SIMULATED]   Max concurrent sequences: 256 (ASSUMED)

[SIMULATED] Running vLLM continuous batching simulation...
[SIMULATED] Simulation complete in 0.01 seconds

Cost Comparison
============================================================

Workload:
  Total requests: 100
  Total input tokens: 50,722
  Total output tokens: 14,592

Self-Hosted GPU (Simulated vLLM on A100 40GB):
[SIMULATED]   Wall clock time: 47.03 seconds
[SIMULATED]   GPU hours: 0.013064
[SIMULATED]   Total cost: $0.0196
[SIMULATED]   Input cost: $0.39 per million tokens
[SIMULATED]   Output cost: $1.34 per million tokens

API Pricing (OpenAI GPT-3.5-turbo):
[MEASURED]   Input cost: $0.50 per million tokens
[MEASURED]   Output cost: $1.50 per million tokens
[MEASURED]   Total cost for this workload: $0.0472

Cost Difference:
[SIMULATED]   Self-hosted saves: $0.0277 (58.5% cheaper)

IMPORTANT:
All [SIMULATED] values use assumed throughput and configuration parameters.
Run 'throttle cost' against a real GPU endpoint for measured costs.
```

## Measure Real Costs

To measure actual costs against a live inference server:

```bash
# First, make sure you have an OpenAI-compatible inference server running
# For example, using Ollama:
# ollama serve &

# Then run cost measurement
throttle cost \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.2:1b \
  --gpu-hourly-rate 1.50 \
  --num-requests 20
```

This will:
- Send 20 test requests to your endpoint
- Measure actual throughput and timing
- Calculate real dollars per million tokens
- Show 95% confidence intervals

## Validate the Simulator

To check how accurate the simulator is for your hardware:

```bash
throttle validate-sim \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.2:1b \
  --gpu-hourly-rate 1.50
```

This compares simulator predictions to real measurements and shows the error percentage.

## Next Steps

- Use the simulator to explore different scenarios without hardware
- Measure actual costs to see real numbers
- Run `throttle validate-sim` to calibrate the simulator for your setup
- For production benchmarks, see `throttle golden --help`

## Getting Help

```bash
# See all available commands
throttle --help

# Get help for a specific command
throttle demo --help
throttle cost --help
```

## Troubleshooting

### "httpx is required for cost measurement"

Install httpx:
```bash
pip install httpx
```

### "Failed to connect to endpoint"

Make sure your inference server is running:
```bash
# For Ollama
ollama serve

# Check it's accessible
curl http://localhost:11434/api/tags
```

### Simulator shows different costs than real measurements

This is expected! The simulator uses assumed throughput values. Run `throttle validate-sim` to see how far off it is, then use `throttle cost` for real measurements.
