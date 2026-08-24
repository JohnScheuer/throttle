"""Pure cost model for GPU inference.

Converts tokens, wall clock time, and GPU hourly rate into dollars per million tokens.
No I/O, no side effects - just arithmetic on measured values.
"""

from typing import NamedTuple


class CostResult(NamedTuple):
    """Cost calculation result with input and output token costs."""
    dollars_per_million_input_tokens: float
    dollars_per_million_output_tokens: float
    total_dollars: float
    gpu_hours: float


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    wall_clock_seconds: float,
    gpu_hourly_rate_dollars: float,
) -> CostResult:
    """Calculate cost per million tokens from measured values.

    Args:
        input_tokens: Total input tokens processed
        output_tokens: Total output tokens generated
        wall_clock_seconds: Wall clock time elapsed
        gpu_hourly_rate_dollars: GPU cost per hour in dollars

    Returns:
        CostResult with per-million-token costs and totals

    Raises:
        ValueError: If any input is negative or if token counts are zero

    Example:
        >>> # 100k input tokens, 50k output tokens, 60 seconds on $1.50/hr GPU
        >>> result = calculate_cost(100_000, 50_000, 60.0, 1.50)
        >>> result.dollars_per_million_input_tokens
        0.25
        >>> result.dollars_per_million_output_tokens
        0.5
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(f"Token counts must be non-negative: input={input_tokens}, output={output_tokens}")

    if wall_clock_seconds < 0:
        raise ValueError(f"Wall clock seconds must be non-negative: {wall_clock_seconds}")

    if gpu_hourly_rate_dollars < 0:
        raise ValueError(f"GPU hourly rate must be non-negative: {gpu_hourly_rate_dollars}")

    if input_tokens == 0 and output_tokens == 0:
        raise ValueError("At least one of input_tokens or output_tokens must be positive")

    # Convert wall clock to GPU hours
    gpu_hours = wall_clock_seconds / 3600.0

    # Total cost = GPU hours * hourly rate
    total_dollars = gpu_hours * gpu_hourly_rate_dollars

    # Cost per million input tokens
    # If input_tokens is 0, set cost to 0 (can't divide by zero)
    if input_tokens > 0:
        dollars_per_million_input = (total_dollars / input_tokens) * 1_000_000
    else:
        dollars_per_million_input = 0.0

    # Cost per million output tokens
    # If output_tokens is 0, set cost to 0
    if output_tokens > 0:
        dollars_per_million_output = (total_dollars / output_tokens) * 1_000_000
    else:
        dollars_per_million_output = 0.0

    return CostResult(
        dollars_per_million_input_tokens=dollars_per_million_input,
        dollars_per_million_output_tokens=dollars_per_million_output,
        total_dollars=total_dollars,
        gpu_hours=gpu_hours,
    )
