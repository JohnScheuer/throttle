"""Test that simulator max_num_seqs parameter affects results under saturation."""
import pytest
from throttle.simulator import VLLMSimulator, SimulatorConfig
from throttle.workload import WorkloadGenerator


def test_max_num_seqs_affects_wall_clock_at_saturation():
    """Simulator produces different wall clock times for different max_num_seqs under saturating load."""
    # Generate a saturating workload: fast arrivals with long outputs
    workload_gen = WorkloadGenerator(seed=42)
    workload = workload_gen.generate_chat_workload(
        num_requests=300,
        arrival_rate_requests_per_sec=30.0,
        mean_prompt_tokens=300,
        mean_output_tokens=1000,
    )
    
    # Test with max_num_seqs=128
    config_128 = SimulatorConfig(
        prefill_throughput_tokens_per_sec=5000.0,
        decode_throughput_tokens_per_sec=100.0,
        max_num_seqs=128,
        saturation_knee_sequences=102,
        kv_cache_capacity_tokens=500_000,
        gpu_hourly_rate_dollars=1.50,
    )
    
    sim_128 = VLLMSimulator(config_128)
    for arrival_time, prompt_tokens, output_tokens in workload:
        sim_128.add_request(arrival_time, prompt_tokens, output_tokens)
    completed_128, wall_clock_128 = sim_128.run()
    
    # Test with max_num_seqs=256
    config_256 = SimulatorConfig(
        prefill_throughput_tokens_per_sec=5000.0,
        decode_throughput_tokens_per_sec=100.0,
        max_num_seqs=256,
        saturation_knee_sequences=200,
        kv_cache_capacity_tokens=500_000,
        gpu_hourly_rate_dollars=1.50,
    )
    
    sim_256 = VLLMSimulator(config_256)
    for arrival_time, prompt_tokens, output_tokens in workload:
        sim_256.add_request(arrival_time, prompt_tokens, output_tokens)
    completed_256, wall_clock_256 = sim_256.run()
    
    # Assert that max_num_seqs=128 takes longer than max_num_seqs=256
    # At this workload (empirically verified: 77.87s vs 62.86s, ~24% difference)
    assert wall_clock_128 > wall_clock_256, (
        f"Expected max_num_seqs=128 to be slower than 256 at saturating load, "
        f"but got 128: {wall_clock_128:.2f}s, 256: {wall_clock_256:.2f}s"
    )
    
    # Verify the difference is meaningful (>10%)
    pct_diff = ((wall_clock_128 - wall_clock_256) / wall_clock_256) * 100
    assert pct_diff > 10.0, (
        f"Expected >10% difference between max_num_seqs=128 and 256, "
        f"but got {pct_diff:.1f}%"
    )
    
    # Both should complete all requests
    assert len(completed_128) == 300
    assert len(completed_256) == 300


def test_max_num_seqs_wired_into_simulator():
    """Verify max_num_seqs is actually used by the simulator (not stubbed/ignored)."""
    # Small workload, but with very low max_num_seqs to force queueing
    workload_gen = WorkloadGenerator(seed=42)
    workload = workload_gen.generate_chat_workload(
        num_requests=50,
        arrival_rate_requests_per_sec=10.0,
        mean_prompt_tokens=200,
        mean_output_tokens=500,
    )
    
    # Test with severely limited max_num_seqs=8
    config_8 = SimulatorConfig(
        prefill_throughput_tokens_per_sec=5000.0,
        decode_throughput_tokens_per_sec=100.0,
        max_num_seqs=8,  # Very small
        saturation_knee_sequences=6,
        kv_cache_capacity_tokens=500_000,
        gpu_hourly_rate_dollars=1.50,
    )
    
    sim_8 = VLLMSimulator(config_8)
    for arrival_time, prompt_tokens, output_tokens in workload:
        sim_8.add_request(arrival_time, prompt_tokens, output_tokens)
    _, wall_clock_8 = sim_8.run()
    
    # Test with comfortable max_num_seqs=128
    config_128 = SimulatorConfig(
        prefill_throughput_tokens_per_sec=5000.0,
        decode_throughput_tokens_per_sec=100.0,
        max_num_seqs=128,  # Plenty of capacity
        saturation_knee_sequences=102,
        kv_cache_capacity_tokens=500_000,
        gpu_hourly_rate_dollars=1.50,
    )
    
    sim_128 = VLLMSimulator(config_128)
    for arrival_time, prompt_tokens, output_tokens in workload:
        sim_128.add_request(arrival_time, prompt_tokens, output_tokens)
    _, wall_clock_128 = sim_128.run()
    
    # max_num_seqs=8 should be MUCH slower due to severe queueing
    assert wall_clock_8 > wall_clock_128 * 1.5, (
        f"Expected max_num_seqs=8 to be at least 50% slower than 128, "
        f"but got 8: {wall_clock_8:.2f}s, 128: {wall_clock_128:.2f}s"
    )
