"""vLLM-like inference simulator using discrete event simulation.

Simulates continuous batching with prefill/decode phases, queueing, saturation,
and KV cache memory pressure. Simulated time advances by arithmetic, not wall clock.
A 300 request workload models in under a second.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EventType(Enum):
    """Event types in the discrete event simulation."""
    REQUEST_ARRIVAL = "request_arrival"
    PREFILL_COMPLETE = "prefill_complete"
    DECODE_STEP = "decode_step"
    REQUEST_COMPLETE = "request_complete"


@dataclass(order=True)
class Event:
    """Discrete event with priority queue ordering by time."""
    time: float
    event_type: EventType = field(compare=False)
    request_id: Optional[int] = field(default=None, compare=False)
    data: Optional[dict] = field(default=None, compare=False)


@dataclass
class Request:
    """Request to be processed."""
    request_id: int
    arrival_time: float
    prompt_tokens: int
    max_new_tokens: int

    # Runtime state
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    tokens_generated: int = 0
    kv_cache_size: int = 0  # Current KV cache usage in tokens


@dataclass
class SimulatorConfig:
    """vLLM simulator configuration parameters.

    All parameters marked ASSUMED until validated against real GPU runs.
    """
    # Prefill throughput (tokens/sec for prefill phase)
    # ASSUMED: Based on A100 40GB estimates, not measured
    prefill_throughput_tokens_per_sec: float = 5000.0

    # Decode throughput (tokens/sec per sequence in decode)
    # ASSUMED: Decode is slower per token than prefill
    decode_throughput_tokens_per_sec: float = 100.0

    # Maximum concurrent sequences (max_num_seqs in vLLM)
    # ASSUMED: Typical value for 7B model on A100 40GB
    max_num_seqs: int = 256

    # Saturation knee: sequences where decode time starts rising
    # ASSUMED: 80% of max_num_seqs
    saturation_knee_sequences: int = 200

    # Saturation penalty: decode time multiplier above knee
    # ASSUMED: Linear degradation from 1.0x at knee to 2.0x at max
    saturation_penalty_at_max: float = 2.0

    # KV cache capacity in tokens
    # ASSUMED: A100 40GB can hold ~500k tokens of KV cache for 7B model
    kv_cache_capacity_tokens: int = 500_000

    # Preemption overhead (seconds to recompute preempted request)
    # ASSUMED: Proportional to prompt length
    preemption_overhead_per_token_sec: float = 0.0002

    # GPU hourly rate in dollars
    # Source: vast.ai A100 40GB spot pricing, 2026-08-24
    # Status: ASSUMED (not yet verified with real rental)
    gpu_hourly_rate_dollars: float = 1.50


class VLLMSimulator:
    """Discrete event simulator for vLLM continuous batching."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.current_time = 0.0
        self.event_queue: List[Event] = []

        # Active requests being processed (prefill or decode)
        self.active_requests: List[Request] = []

        # Queue for requests waiting for available slots
        self.waiting_queue: List[Request] = []

        # Completed requests
        self.completed_requests: List[Request] = []

        # Preempted requests (need recomputation)
        self.preempted_requests: List[Request] = []

        # KV cache usage tracking
        self.current_kv_cache_usage = 0

        # Next request ID
        self.next_request_id = 0

        # Track if decode step is already scheduled
        self.decode_step_scheduled = False
        self.next_decode_time = None

    def schedule_event(self, event: Event):
        """Add event to priority queue."""
        heapq.heappush(self.event_queue, event)

    def add_request(self, arrival_time: float, prompt_tokens: int, max_new_tokens: int) -> Request:
        """Add a request to the workload."""
        request = Request(
            request_id=self.next_request_id,
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )
        self.next_request_id += 1

        # Schedule arrival event
        self.schedule_event(Event(
            time=arrival_time,
            event_type=EventType.REQUEST_ARRIVAL,
            request_id=request.request_id,
            data={'request': request}
        ))

        return request

    def can_admit_request(self) -> bool:
        """Check if we can admit a new request from the waiting queue."""
        return len(self.active_requests) < self.config.max_num_seqs

    def get_decode_time_per_token(self, num_active: int) -> float:
        """Calculate decode time per token based on current batch size and saturation."""
        base_time = 1.0 / self.config.decode_throughput_tokens_per_sec

        # Apply saturation penalty if above knee
        if num_active > self.config.saturation_knee_sequences:
            # Linear interpolation from 1.0x at knee to penalty at max
            fraction_above_knee = (num_active - self.config.saturation_knee_sequences) / \
                                 (self.config.max_num_seqs - self.config.saturation_knee_sequences)
            penalty = 1.0 + (self.config.saturation_penalty_at_max - 1.0) * fraction_above_knee
            return base_time * penalty

        return base_time

    def check_kv_cache_pressure(self):
        """Check KV cache usage and preempt if necessary."""
        if self.current_kv_cache_usage > self.config.kv_cache_capacity_tokens:
            # Preempt oldest requests until we're under capacity
            # Sort by prefill start time (oldest first)
            candidates = sorted(self.active_requests, key=lambda r: r.prefill_start_time or 0)

            for req in candidates:
                if self.current_kv_cache_usage <= self.config.kv_cache_capacity_tokens:
                    break

                # Preempt this request
                self.active_requests.remove(req)
                self.preempted_requests.append(req)
                self.current_kv_cache_usage -= req.kv_cache_size
                req.kv_cache_size = 0  # Will need to recompute

    def handle_request_arrival(self, event: Event):
        """Handle a request arriving at the system."""
        request = event.data['request']

        if self.can_admit_request():
            # Admit immediately and start prefill
            self.active_requests.append(request)
            request.prefill_start_time = self.current_time

            # Prefill time = prompt_tokens / throughput
            prefill_duration = request.prompt_tokens / self.config.prefill_throughput_tokens_per_sec

            # Schedule prefill completion
            self.schedule_event(Event(
                time=self.current_time + prefill_duration,
                event_type=EventType.PREFILL_COMPLETE,
                request_id=request.request_id,
            ))

            # Reserve KV cache for prompt
            request.kv_cache_size = request.prompt_tokens
            self.current_kv_cache_usage += request.prompt_tokens
            self.check_kv_cache_pressure()
        else:
            # Queue the request
            self.waiting_queue.append(request)

    def handle_prefill_complete(self, event: Event):
        """Handle prefill phase completion, start decode."""
        request = next((r for r in self.active_requests if r.request_id == event.request_id), None)
        if not request:
            return  # Request was preempted

        request.prefill_end_time = self.current_time

        # If no output tokens requested, complete immediately
        if request.max_new_tokens == 0:
            request.completion_time = self.current_time
            self.active_requests.remove(request)
            self.completed_requests.append(request)
            self.current_kv_cache_usage -= request.kv_cache_size
            return

        request.decode_start_time = self.current_time

        # Schedule first decode step
        self.schedule_decode_step()

    def schedule_decode_step(self):
        """Schedule the next decode step for all active decoding requests."""
        # Find all requests in decode phase
        decoding = [r for r in self.active_requests if r.decode_start_time is not None and r.completion_time is None]

        if not decoding:
            self.decode_step_scheduled = False
            return

        # If there's already a decode step scheduled, don't schedule another
        # The existing one will handle all currently decoding requests
        if self.decode_step_scheduled:
            return

        # Decode time based on current batch size
        decode_time = self.get_decode_time_per_token(len(self.active_requests))
        next_time = self.current_time + decode_time

        # Schedule decode step for next token
        self.schedule_event(Event(
            time=next_time,
            event_type=EventType.DECODE_STEP,
        ))
        self.decode_step_scheduled = True
        self.next_decode_time = next_time

    def handle_decode_step(self, event: Event):
        """Handle one decode step - generate one token for all active sequences."""
        # Clear the scheduled flag so new decode steps can be scheduled
        self.decode_step_scheduled = False
        self.next_decode_time = None

        # All active decoding requests generate one token
        completed_this_step = []

        for request in self.active_requests:
            if request.decode_start_time is None or request.completion_time is not None:
                continue  # Not in decode phase yet or already complete

            # Generate one token
            request.tokens_generated += 1
            request.kv_cache_size += 1  # Add generated token to KV cache
            self.current_kv_cache_usage += 1

            # Check if request is complete
            if request.tokens_generated >= request.max_new_tokens:
                request.completion_time = self.current_time
                completed_this_step.append(request)

        # Move completed requests
        for request in completed_this_step:
            self.active_requests.remove(request)
            self.completed_requests.append(request)
            self.current_kv_cache_usage -= request.kv_cache_size

        # Admit waiting requests if slots available
        while self.waiting_queue and self.can_admit_request():
            waiting_req = self.waiting_queue.pop(0)

            # Start prefill
            self.active_requests.append(waiting_req)
            waiting_req.prefill_start_time = self.current_time

            prefill_duration = waiting_req.prompt_tokens / self.config.prefill_throughput_tokens_per_sec

            self.schedule_event(Event(
                time=self.current_time + prefill_duration,
                event_type=EventType.PREFILL_COMPLETE,
                request_id=waiting_req.request_id,
            ))

            waiting_req.kv_cache_size = waiting_req.prompt_tokens
            self.current_kv_cache_usage += waiting_req.prompt_tokens

        self.check_kv_cache_pressure()

        # Schedule next decode step if there are still active requests
        if any(r.decode_start_time is not None and r.completion_time is None for r in self.active_requests):
            self.schedule_decode_step()

    def run(self) -> Tuple[List[Request], float]:
        """Run simulation until all requests complete.

        Returns:
            (completed_requests, total_wall_clock_time)
        """
        while self.event_queue or self.active_requests or self.waiting_queue:
            if not self.event_queue:
                break

            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == EventType.REQUEST_ARRIVAL:
                self.handle_request_arrival(event)
            elif event.event_type == EventType.PREFILL_COMPLETE:
                self.handle_prefill_complete(event)
            elif event.event_type == EventType.DECODE_STEP:
                self.handle_decode_step(event)

        return self.completed_requests, self.current_time
