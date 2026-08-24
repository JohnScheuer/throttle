"""Workload generation for simulator benchmarks.

Generates realistic request patterns for LLM inference workloads.
"""

import random
from typing import List, Tuple


class WorkloadGenerator:
    """Generate synthetic workloads for LLM inference simulation."""

    def __init__(self, seed: int = 42):
        """Initialize workload generator with random seed for reproducibility.

        Args:
            seed: Random seed for reproducible workloads
        """
        self.rng = random.Random(seed)

    def generate_chat_workload(
        self,
        num_requests: int,
        arrival_rate_requests_per_sec: float = 1.0,
        mean_prompt_tokens: int = 500,
        mean_output_tokens: int = 150,
    ) -> List[Tuple[float, int, int]]:
        """Generate a chat-like workload with varied prompt/output lengths.

        Models typical chat usage where:
        - Prompts vary from short questions to long context
        - Outputs vary from brief answers to detailed responses
        - Requests arrive following a Poisson process

        Args:
            num_requests: Number of requests to generate
            arrival_rate_requests_per_sec: Average requests per second
            mean_prompt_tokens: Mean prompt length in tokens
            mean_output_tokens: Mean output length in tokens

        Returns:
            List of (arrival_time, prompt_tokens, max_new_tokens) tuples
        """
        workload = []
        current_time = 0.0

        for _ in range(num_requests):
            # Poisson arrival process (exponential inter-arrival times)
            interarrival = self.rng.expovariate(arrival_rate_requests_per_sec)
            current_time += interarrival

            # Log-normal distribution for token counts (realistic skew)
            # Most requests are short, some are very long
            prompt_tokens = max(10, int(self.rng.lognormvariate(
                mu=self._log_mean(mean_prompt_tokens, 0.8),
                sigma=0.8
            )))

            output_tokens = max(1, int(self.rng.lognormvariate(
                mu=self._log_mean(mean_output_tokens, 0.6),
                sigma=0.6
            )))

            workload.append((current_time, prompt_tokens, output_tokens))

        return workload

    def generate_burst_workload(
        self,
        num_requests: int,
        burst_size: int = 10,
        burst_interval_sec: float = 5.0,
        mean_prompt_tokens: int = 500,
        mean_output_tokens: int = 150,
    ) -> List[Tuple[float, int, int]]:
        """Generate a bursty workload with clusters of simultaneous requests.

        Models scenarios like:
        - Batch processing jobs
        - Traffic spikes after content publication
        - Synchronization points in distributed systems

        Args:
            num_requests: Total number of requests
            burst_size: Number of requests per burst
            burst_interval_sec: Time between bursts
            mean_prompt_tokens: Mean prompt length
            mean_output_tokens: Mean output length

        Returns:
            List of (arrival_time, prompt_tokens, max_new_tokens) tuples
        """
        workload = []
        current_time = 0.0

        while len(workload) < num_requests:
            # Generate a burst of requests
            for _ in range(min(burst_size, num_requests - len(workload))):
                # Small jitter within the burst (all arrive ~same time)
                jitter = self.rng.uniform(0, 0.1)

                prompt_tokens = max(10, int(self.rng.lognormvariate(
                    mu=self._log_mean(mean_prompt_tokens, 0.8),
                    sigma=0.8
                )))

                output_tokens = max(1, int(self.rng.lognormvariate(
                    mu=self._log_mean(mean_output_tokens, 0.6),
                    sigma=0.6
                )))

                workload.append((current_time + jitter, prompt_tokens, output_tokens))

            # Wait for next burst
            current_time += burst_interval_sec

        # Sort by arrival time (jitter may have shuffled order)
        workload.sort(key=lambda x: x[0])

        return workload[:num_requests]

    def _log_mean(self, mean: float, sigma: float) -> float:
        """Calculate mu for lognormal distribution to achieve desired mean.

        For lognormal(mu, sigma), the mean is exp(mu + sigma^2/2).
        Solving for mu: mu = log(mean) - sigma^2/2
        """
        import math
        return math.log(mean) - (sigma ** 2) / 2
