"""Unit tests for cost_model module.

Tests against hand-computed values to ensure arithmetic is correct.
"""

import pytest
from throttle.cost_model import calculate_cost, CostResult


class TestCostModel:
    """Test cost calculations against hand-computed values."""

    def test_basic_calculation(self):
        """
        Hand computation:
        - 100k input tokens, 50k output tokens
        - 60 seconds = 1/60 hour = 0.01667 hours
        - $1.50/hour GPU
        - Total cost = 0.01667 * 1.50 = $0.025
        - Input cost per M = (0.025 / 100k) * 1M = $0.25/M
        - Output cost per M = (0.025 / 50k) * 1M = $0.50/M
        """
        result = calculate_cost(
            input_tokens=100_000,
            output_tokens=50_000,
            wall_clock_seconds=60.0,
            gpu_hourly_rate_dollars=1.50,
        )

        assert abs(result.gpu_hours - 0.01667) < 0.00001
        assert abs(result.total_dollars - 0.025) < 0.0001
        assert abs(result.dollars_per_million_input_tokens - 0.25) < 0.01
        assert abs(result.dollars_per_million_output_tokens - 0.50) < 0.01

    def test_one_hour_at_one_dollar(self):
        """
        Hand computation:
        - 1M input, 1M output
        - 3600 seconds = 1 hour
        - $1.00/hour
        - Total cost = 1.0 * 1.0 = $1.00
        - Input cost per M = (1.0 / 1M) * 1M = $1.00/M
        - Output cost per M = (1.0 / 1M) * 1M = $1.00/M
        """
        result = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_clock_seconds=3600.0,
            gpu_hourly_rate_dollars=1.00,
        )

        assert result.gpu_hours == 1.0
        assert result.total_dollars == 1.0
        assert result.dollars_per_million_input_tokens == 1.0
        assert result.dollars_per_million_output_tokens == 1.0

    def test_expensive_gpu(self):
        """
        Hand computation:
        - 500k input, 500k output
        - 1800 seconds = 0.5 hours
        - $10.00/hour (expensive GPU)
        - Total cost = 0.5 * 10.0 = $5.00
        - Input cost per M = (5.0 / 500k) * 1M = $10.00/M
        - Output cost per M = (5.0 / 500k) * 1M = $10.00/M
        """
        result = calculate_cost(
            input_tokens=500_000,
            output_tokens=500_000,
            wall_clock_seconds=1800.0,
            gpu_hourly_rate_dollars=10.00,
        )

        assert result.gpu_hours == 0.5
        assert result.total_dollars == 5.0
        assert result.dollars_per_million_input_tokens == 10.0
        assert result.dollars_per_million_output_tokens == 10.0

    def test_asymmetric_tokens(self):
        """
        Hand computation:
        - 1M input, 100k output (10:1 ratio)
        - 600 seconds = 1/6 hour = 0.1667 hours
        - $3.00/hour
        - Total cost = 0.1667 * 3.0 = $0.50
        - Input cost per M = (0.50 / 1M) * 1M = $0.50/M
        - Output cost per M = (0.50 / 100k) * 1M = $5.00/M
        """
        result = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=100_000,
            wall_clock_seconds=600.0,
            gpu_hourly_rate_dollars=3.00,
        )

        assert abs(result.gpu_hours - 0.1667) < 0.0001
        assert abs(result.total_dollars - 0.50) < 0.01
        assert abs(result.dollars_per_million_input_tokens - 0.50) < 0.01
        assert abs(result.dollars_per_million_output_tokens - 5.00) < 0.01

    def test_zero_input_tokens(self):
        """
        Edge case: only output tokens (e.g., continuation from cache).
        Hand computation:
        - 0 input, 100k output
        - 30 seconds = 1/120 hour = 0.00833 hours
        - $2.00/hour
        - Total cost = 0.00833 * 2.0 = $0.01667
        - Input cost per M = $0.00/M (no input tokens)
        - Output cost per M = (0.01667 / 100k) * 1M = $0.1667/M
        """
        result = calculate_cost(
            input_tokens=0,
            output_tokens=100_000,
            wall_clock_seconds=30.0,
            gpu_hourly_rate_dollars=2.00,
        )

        assert abs(result.gpu_hours - 0.00833) < 0.00001
        assert abs(result.total_dollars - 0.01667) < 0.0001
        assert result.dollars_per_million_input_tokens == 0.0
        assert abs(result.dollars_per_million_output_tokens - 0.1667) < 0.001

    def test_zero_output_tokens(self):
        """
        Edge case: only input tokens (e.g., embedding generation).
        Hand computation:
        - 100k input, 0 output
        - 30 seconds = 1/120 hour
        - $2.00/hour
        - Total cost = (1/120) * 2.0 = $0.01667
        - Input cost per M = (0.01667 / 100k) * 1M = $0.1667/M
        - Output cost per M = $0.00/M (no output tokens)
        """
        result = calculate_cost(
            input_tokens=100_000,
            output_tokens=0,
            wall_clock_seconds=30.0,
            gpu_hourly_rate_dollars=2.00,
        )

        assert abs(result.dollars_per_million_input_tokens - 0.1667) < 0.001
        assert result.dollars_per_million_output_tokens == 0.0

    def test_small_numbers(self):
        """
        Test with small token counts to verify precision.
        Hand computation:
        - 100 input, 50 output
        - 1 second = 1/3600 hour = 0.000278 hours
        - $1.00/hour
        - Total cost = 0.000278 * 1.0 = $0.000278
        - Input cost per M = (0.000278 / 100) * 1M = $2.78/M
        - Output cost per M = (0.000278 / 50) * 1M = $5.56/M
        """
        result = calculate_cost(
            input_tokens=100,
            output_tokens=50,
            wall_clock_seconds=1.0,
            gpu_hourly_rate_dollars=1.00,
        )

        assert abs(result.dollars_per_million_input_tokens - 2.78) < 0.01
        assert abs(result.dollars_per_million_output_tokens - 5.56) < 0.01

    def test_free_gpu(self):
        """
        Test with zero GPU cost (e.g., free tier).
        """
        result = calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            wall_clock_seconds=3600.0,
            gpu_hourly_rate_dollars=0.0,
        )

        assert result.total_dollars == 0.0
        assert result.dollars_per_million_input_tokens == 0.0
        assert result.dollars_per_million_output_tokens == 0.0

    def test_validation_negative_input_tokens(self):
        """Test that negative input tokens raise ValueError."""
        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            calculate_cost(-100, 100, 60.0, 1.0)

    def test_validation_negative_output_tokens(self):
        """Test that negative output tokens raise ValueError."""
        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            calculate_cost(100, -100, 60.0, 1.0)

    def test_validation_negative_time(self):
        """Test that negative wall clock raises ValueError."""
        with pytest.raises(ValueError, match="Wall clock seconds must be non-negative"):
            calculate_cost(100, 100, -60.0, 1.0)

    def test_validation_negative_gpu_rate(self):
        """Test that negative GPU rate raises ValueError."""
        with pytest.raises(ValueError, match="GPU hourly rate must be non-negative"):
            calculate_cost(100, 100, 60.0, -1.0)

    def test_validation_zero_tokens(self):
        """Test that zero input and output tokens raise ValueError."""
        with pytest.raises(ValueError, match="At least one of input_tokens or output_tokens must be positive"):
            calculate_cost(0, 0, 60.0, 1.0)

    def test_result_is_namedtuple(self):
        """Verify CostResult has expected fields."""
        result = calculate_cost(100_000, 50_000, 60.0, 1.50)

        assert hasattr(result, 'dollars_per_million_input_tokens')
        assert hasattr(result, 'dollars_per_million_output_tokens')
        assert hasattr(result, 'total_dollars')
        assert hasattr(result, 'gpu_hours')

        # Verify it's a namedtuple (can access by index and by name)
        assert result[0] == result.dollars_per_million_input_tokens
        assert result[1] == result.dollars_per_million_output_tokens
        assert result[2] == result.total_dollars
        assert result[3] == result.gpu_hours
