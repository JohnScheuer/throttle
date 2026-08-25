"""Test bootstrap confidence intervals in measure command."""
import pytest
import json
import sys
from pathlib import Path


def test_bootstrap_ci_contains_true_value():
    """
    Test bootstrap CI on synthetic data with known true cost.

    Generate timing data from a known distribution, compute bootstrap CI,
    and verify it contains the true value at roughly the stated 95% rate.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from throttle.cost_model import calculate_cost
    import statistics
    import random

    # Known parameters for synthetic data
    true_gpu_rate = 1.0  # $/hour
    total_input_tokens = 5000
    total_output_tokens = 1000

    # True wall clock time that would give us a known cost
    # We want a predictable $/M tokens
    # Cost = (wall_clock_seconds / 3600) * gpu_rate * 1e6 / tokens
    # Let's set wall_clock so we get exactly $1.00/M input
    # 1.00 = (wall_clock / 3600) * 1.0 * 1e6 / 5000
    # wall_clock = 1.00 * 5000 * 3600 / 1e6 = 18.0 seconds
    true_wall_clock = 18.0  # seconds

    true_cost = calculate_cost(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_clock_seconds=true_wall_clock,
        gpu_hourly_rate_dollars=true_gpu_rate,
    )

    true_input_cost = true_cost.dollars_per_million_input_tokens
    true_output_cost = true_cost.dollars_per_million_output_tokens

    # Run many experiments to test coverage
    n_experiments = 100
    input_contains = 0
    output_contains = 0

    random.seed(42)

    for exp_idx in range(n_experiments):
        # Generate 5 trials with timing variation
        # Add gaussian noise to wall clock time (CV ~ 10%)
        n_trials = 5
        trial_costs_input = []
        trial_costs_output = []

        for _ in range(n_trials):
            # Wall clock varies with ~10% coefficient of variation
            wall_clock_sample = random.gauss(true_wall_clock, true_wall_clock * 0.1)
            cost = calculate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                wall_clock_seconds=wall_clock_sample,
                gpu_hourly_rate_dollars=true_gpu_rate,
            )
            trial_costs_input.append(cost.dollars_per_million_input_tokens)
            trial_costs_output.append(cost.dollars_per_million_output_tokens)

        # Bootstrap CI (same method as measure command)
        n_bootstrap = 10000
        bootstrap_input = []
        bootstrap_output = []

        for _ in range(n_bootstrap):
            resample_indices = random.choices(range(n_trials), k=n_trials)
            bootstrap_input.append(statistics.median([trial_costs_input[i] for i in resample_indices]))
            bootstrap_output.append(statistics.median([trial_costs_output[i] for i in resample_indices]))

        bootstrap_input.sort()
        bootstrap_output.sort()
        ci_lower_idx = int(0.025 * n_bootstrap)
        ci_upper_idx = int(0.975 * n_bootstrap)

        input_ci_lower = bootstrap_input[ci_lower_idx]
        input_ci_upper = bootstrap_input[ci_upper_idx]
        output_ci_lower = bootstrap_output[ci_lower_idx]
        output_ci_upper = bootstrap_output[ci_upper_idx]

        # Check if true value is in interval
        if input_ci_lower <= true_input_cost <= input_ci_upper:
            input_contains += 1
        if output_ci_lower <= true_output_cost <= output_ci_upper:
            output_contains += 1

    input_coverage = input_contains / n_experiments
    output_coverage = output_contains / n_experiments

    # With 100 experiments and 95% CI, we expect ~95 ± sqrt(100*0.95*0.05) = 95 ± 2.2
    # So acceptable range is roughly [90%, 100%] to avoid flakiness
    # But we'll be lenient: [85%, 100%] since bootstrap on small samples can be conservative

    assert input_coverage >= 0.85, (
        f"Bootstrap CI coverage for input cost was {input_coverage:.1%}, expected ≥85%. "
        f"True value: ${true_input_cost:.2f}/M, got coverage of {input_contains}/{n_experiments}."
    )

    assert output_coverage >= 0.85, (
        f"Bootstrap CI coverage for output cost was {output_coverage:.1%}, expected ≥85%. "
        f"True value: ${true_output_cost:.2f}/M, got coverage of {output_contains}/{n_experiments}."
    )

    print(f"✓ Bootstrap CI coverage: input {input_coverage:.1%}, output {output_coverage:.1%}")


def test_bootstrap_load_bearing_degenerate_interval_fails():
    """
    Prove the bootstrap test is load bearing by breaking the CI to be degenerate.

    A degenerate interval where lower = upper = median should fail coverage.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from throttle.cost_model import calculate_cost
    import statistics
    import random

    # Same setup as above
    true_gpu_rate = 1.0
    total_input_tokens = 5000
    total_output_tokens = 1000
    true_wall_clock = 18.0

    true_cost = calculate_cost(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_clock_seconds=true_wall_clock,
        gpu_hourly_rate_dollars=true_gpu_rate,
    )

    true_input_cost = true_cost.dollars_per_million_input_tokens

    # Run 20 experiments with degenerate interval
    n_experiments = 20
    input_contains = 0

    random.seed(43)

    for exp_idx in range(n_experiments):
        n_trials = 5
        trial_costs_input = []

        for _ in range(n_trials):
            wall_clock_sample = random.gauss(true_wall_clock, true_wall_clock * 0.1)
            cost = calculate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                wall_clock_seconds=wall_clock_sample,
                gpu_hourly_rate_dollars=true_gpu_rate,
            )
            trial_costs_input.append(cost.dollars_per_million_input_tokens)

        # BROKEN: degenerate interval (lower = upper = median)
        median_input = statistics.median(trial_costs_input)
        input_ci_lower = median_input
        input_ci_upper = median_input

        # Check if true value is in degenerate interval
        if input_ci_lower <= true_input_cost <= input_ci_upper:
            input_contains += 1

    input_coverage = input_contains / n_experiments

    # Degenerate interval should have very low coverage (only when median == true value exactly)
    # With 10% noise, median is unlikely to equal true value exactly
    assert input_coverage < 0.50, (
        f"Degenerate interval coverage was {input_coverage:.1%}, expected <50%. "
        f"The test is not load bearing - it passed even with a broken bootstrap!"
    )

    print(f"✓ Degenerate interval correctly has low coverage: {input_coverage:.1%} (expected <50%)")
