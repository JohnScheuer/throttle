"""Validation tests for simulator with hand-computed expected answers.

Each test has a fixed workload and config where expected results are computed
by hand to verify the simulator produces correct output.
"""

import pytest
from throttle.simulator import VLLMSimulator, SimulatorConfig
from throttle.cost_model import calculate_cost


class TestSimulatorValidation:
    """Validation tests against hand-computed expected values."""

    def test_two_requests_parallel_no_saturation(self):
        """
        Hand computation:
        - 2 requests, both arrive at t=0
        - Request 1: 100 prompt tokens, 10 output tokens
        - Request 2: 100 prompt tokens, 10 output tokens
        - Config: prefill 1000 tok/sec, decode 100 tok/sec, max_num_seqs=2

        Expected timeline:
        - t=0: Both requests arrive, both admitted (2 slots available)
        - t=0 to t=0.1: Both prefill (100/1000 = 0.1 sec each)
        - t=0.1: Both prefills complete, both start decode
        - t=0.1 to t=0.2: Both decode 10 tokens (10 steps * 0.01 sec/step = 0.1 sec)
        - t=0.2: Both complete

        Expected totals:
        - Wall clock: 0.2 sec
        - Total input tokens: 200
        - Total output tokens: 20
        - GPU hours: 0.2/3600 = 0.0000556 hours
        - At $1.50/hr: $0.0000833
        - Cost per M input: ($0.0000833 / 200) * 1M = $0.417
        - Cost per M output: ($0.0000833 / 20) * 1M = $4.17
        """
        config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=1000.0,
            decode_throughput_tokens_per_sec=100.0,
            max_num_seqs=2,
            saturation_knee_sequences=2,  # No saturation for this test
            gpu_hourly_rate_dollars=1.50,
        )

        sim = VLLMSimulator(config)

        # Add two requests at t=0
        sim.add_request(arrival_time=0.0, prompt_tokens=100, max_new_tokens=10)
        sim.add_request(arrival_time=0.0, prompt_tokens=100, max_new_tokens=10)

        # Run simulation
        completed, wall_clock = sim.run()

        # Verify timeline
        assert len(completed) == 2
        assert abs(wall_clock - 0.2) < 0.001, f"Expected 0.2 sec, got {wall_clock}"

        # Verify request timing
        for req in completed:
            assert abs(req.prefill_start_time - 0.0) < 0.001
            assert abs(req.prefill_end_time - 0.1) < 0.001
            assert abs(req.completion_time - 0.2) < 0.001
            assert req.tokens_generated == 10

        # Verify cost calculation
        total_input = sum(r.prompt_tokens for r in completed)
        total_output = sum(r.tokens_generated for r in completed)

        assert total_input == 200
        assert total_output == 20

        cost_result = calculate_cost(
            input_tokens=total_input,
            output_tokens=total_output,
            wall_clock_seconds=wall_clock,
            gpu_hourly_rate_dollars=config.gpu_hourly_rate_dollars,
        )

        assert abs(cost_result.gpu_hours - 0.0000556) < 0.0000001
        assert abs(cost_result.total_dollars - 0.0000833) < 0.000001
        assert abs(cost_result.dollars_per_million_input_tokens - 0.417) < 0.001
        assert abs(cost_result.dollars_per_million_output_tokens - 4.17) < 0.01

    def test_sequential_requests_with_queueing(self):
        """
        Hand computation:
        - 2 requests, both arrive at t=0
        - Request 1: 100 prompt tokens, 5 output tokens
        - Request 2: 100 prompt tokens, 5 output tokens
        - Config: prefill 1000 tok/sec, decode 100 tok/sec, max_num_seqs=1 (force queueing)

        Expected timeline:
        - t=0: Request 1 arrives and admitted, Request 2 queued
        - t=0 to t=0.1: Request 1 prefill
        - t=0.1: Request 1 starts decode
        - t=0.1 to t=0.15: Request 1 decodes 5 tokens (5 * 0.01 = 0.05 sec)
        - t=0.15: Request 1 completes, Request 2 admitted from queue
        - t=0.15 to t=0.25: Request 2 prefill
        - t=0.25: Request 2 starts decode
        - t=0.25 to t=0.30: Request 2 decodes 5 tokens
        - t=0.30: Request 2 completes

        Expected totals:
        - Wall clock: 0.30 sec
        - Total input tokens: 200
        - Total output tokens: 10
        - Cost per M input: ($1.50 * 0.30/3600 / 200) * 1M = $0.625
        - Cost per M output: ($1.50 * 0.30/3600 / 10) * 1M = $12.50
        """
        config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=1000.0,
            decode_throughput_tokens_per_sec=100.0,
            max_num_seqs=1,  # Force sequential processing
            gpu_hourly_rate_dollars=1.50,
        )

        sim = VLLMSimulator(config)

        sim.add_request(arrival_time=0.0, prompt_tokens=100, max_new_tokens=5)
        sim.add_request(arrival_time=0.0, prompt_tokens=100, max_new_tokens=5)

        completed, wall_clock = sim.run()

        assert len(completed) == 2
        assert abs(wall_clock - 0.30) < 0.001, f"Expected 0.30 sec, got {wall_clock}"

        # Request 1 timeline
        req1 = completed[0]
        assert abs(req1.prefill_start_time - 0.0) < 0.001
        assert abs(req1.prefill_end_time - 0.1) < 0.001
        assert abs(req1.completion_time - 0.15) < 0.001

        # Request 2 timeline (queued, starts after req1 completes)
        req2 = completed[1]
        assert abs(req2.prefill_start_time - 0.15) < 0.001
        assert abs(req2.prefill_end_time - 0.25) < 0.001
        assert abs(req2.completion_time - 0.30) < 0.001

        # Verify cost
        cost_result = calculate_cost(
            input_tokens=200,
            output_tokens=10,
            wall_clock_seconds=wall_clock,
            gpu_hourly_rate_dollars=1.50,
        )

        assert abs(cost_result.dollars_per_million_input_tokens - 0.625) < 0.001
        assert abs(cost_result.dollars_per_million_output_tokens - 12.50) < 0.01

    def test_single_request_baseline(self):
        """
        Simplest possible case: one request, no queueing, no saturation.

        Hand computation:
        - 1 request: 1000 prompt tokens, 100 output tokens
        - Prefill: 1000/5000 = 0.2 sec
        - Decode: 100 * (1/100) = 1.0 sec
        - Total: 1.2 sec
        - Cost at $1.50/hr: $1.50 * 1.2/3600 = $0.0005
        - Input cost per M: ($0.0005 / 1000) * 1M = $0.50
        - Output cost per M: ($0.0005 / 100) * 1M = $5.00
        """
        config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=5000.0,
            decode_throughput_tokens_per_sec=100.0,
            gpu_hourly_rate_dollars=1.50,
        )

        sim = VLLMSimulator(config)
        sim.add_request(arrival_time=0.0, prompt_tokens=1000, max_new_tokens=100)

        completed, wall_clock = sim.run()

        assert len(completed) == 1
        assert abs(wall_clock - 1.2) < 0.001, f"Expected 1.2 sec, got {wall_clock}"

        req = completed[0]
        assert abs(req.prefill_end_time - 0.2) < 0.001
        assert abs(req.completion_time - 1.2) < 0.001
        assert req.tokens_generated == 100

        cost_result = calculate_cost(
            input_tokens=1000,
            output_tokens=100,
            wall_clock_seconds=wall_clock,
            gpu_hourly_rate_dollars=1.50,
        )

        assert abs(cost_result.dollars_per_million_input_tokens - 0.50) < 0.01
        assert abs(cost_result.dollars_per_million_output_tokens - 5.00) < 0.01

    def test_staggered_arrivals(self):
        """
        Test with requests arriving at different times.

        Hand computation:
        - Request 1: arrives t=0, 100 tokens prompt, 10 output
        - Request 2: arrives t=0.05, 100 tokens prompt, 10 output
        - Config: prefill 1000 tok/sec, decode 100 tok/sec

        Timeline:
        - t=0: Request 1 arrives, prefill starts
        - t=0.05: Request 2 arrives, prefill starts (both can run)
        - t=0.1: Request 1 prefill completes, decode starts
        - t=0.11: R1 token 1
        - t=0.12: R1 token 2
        - t=0.13: R1 token 3
        - t=0.14: R1 token 4
        - t=0.15: Request 2 prefill completes, joins decode batch
        - t=0.15: R1 token 5, R2 token 1 (both decode together)
        - t=0.16: R1 token 6, R2 token 2
        - t=0.17: R1 token 7, R2 token 3
        - t=0.18: R1 token 8, R2 token 4
        - t=0.19: R1 token 9, R2 token 5
        - t=0.20: R1 token 10, R2 token 6 → R1 completes
        - t=0.21: R2 token 7
        - t=0.22: R2 token 8
        - t=0.23: R2 token 9
        - t=0.24: R2 token 10 → R2 completes

        Total wall clock: 0.24 sec
        """
        config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=1000.0,
            decode_throughput_tokens_per_sec=100.0,
            max_num_seqs=2,
        )

        sim = VLLMSimulator(config)
        sim.add_request(arrival_time=0.0, prompt_tokens=100, max_new_tokens=10)
        sim.add_request(arrival_time=0.05, prompt_tokens=100, max_new_tokens=10)

        completed, wall_clock = sim.run()

        assert len(completed) == 2

        # Request 1 completes first
        req1 = [r for r in completed if r.request_id == 0][0]
        assert abs(req1.completion_time - 0.2) < 0.001, f"Expected 0.2, got {req1.completion_time}"

        # Request 2 completes after
        req2 = [r for r in completed if r.request_id == 1][0]
        assert abs(req2.completion_time - 0.24) < 0.001, f"Expected 0.24, got {req2.completion_time}"

        assert abs(wall_clock - 0.24) < 0.001

    def test_empty_workload(self):
        """Test simulator with no requests."""
        config = SimulatorConfig()
        sim = VLLMSimulator(config)

        completed, wall_clock = sim.run()

        assert len(completed) == 0
        assert wall_clock == 0.0

    def test_zero_output_tokens(self):
        """
        Test request with zero output tokens (prompt-only, like embeddings).

        Hand computation:
        - 1 request: 100 prompt tokens, 0 output tokens
        - Prefill only: 100/1000 = 0.1 sec
        - Total: 0.1 sec
        """
        config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=1000.0,
        )

        sim = VLLMSimulator(config)
        sim.add_request(arrival_time=0.0, prompt_tokens=100, max_new_tokens=0)

        completed, wall_clock = sim.run()

        assert len(completed) == 1
        assert abs(wall_clock - 0.1) < 0.001
        assert completed[0].tokens_generated == 0
