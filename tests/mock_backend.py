"""Mock LLM backend for cache benchmarking.

Simulates realistic backend latency without requiring a real inference server.
Returns OpenAI-compatible chat completion responses.
"""

import random
import time
from typing import Any


# Knowledge base for deterministic responses
_KNOWLEDGE_BASE: dict[str, str] = {
    "capital of france": "The capital of France is Paris.",
    "capital of japan": "The capital of Japan is Tokyo.",
    "capital of india": "The capital of India is New Delhi.",
    "largest planet": "Jupiter is the largest planet in the solar system.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792 km/s.",
    "reset password": "To reset your password, go to Settings > Security > Reset Password.",
    "cancel subscription": "You can cancel your subscription from Account > Billing > Cancel Plan.",
    "refund policy": "Refunds are issued within 5-7 business days for eligible purchases.",
    "python list vs tuple": "Lists are mutable and use [], tuples are immutable and use ().",
    "what is machine learning": "Machine learning is a field of AI where models learn patterns from data.",
    "what is a neural network": "A neural network is a computational model inspired by biological neurons.",
    "how does tcp work": "TCP establishes a reliable, ordered, connection-oriented byte stream via a handshake.",
}


def _find_match(prompt: str) -> str | None:
    """Find knowledge base key contained in prompt."""
    lowered = prompt.lower()
    for key in _KNOWLEDGE_BASE:
        if key in lowered:
            return key
    return None


def mock_chat_completion(prompt: str, simulate_latency: bool = True) -> dict[str, Any]:
    """Generate OpenAI-compatible chat completion response.

    Latency model: Gaussian ~180ms (realistic LLM API latency) with 5% slow tail.
    """
    if simulate_latency:
        # Base latency: mean 180ms, stddev 50ms
        base_latency = random.gauss(0.18, 0.05)
        base_latency = max(0.05, base_latency)  # Floor at 50ms

        # 5% of requests are slow (add 200-400ms)
        if random.random() < 0.05:
            base_latency += random.uniform(0.2, 0.4)

        time.sleep(base_latency)

    # Find response from knowledge base or synthesize
    key = _find_match(prompt)
    content = _KNOWLEDGE_BASE[key] if key else f"Here is information related to: '{prompt}'. (synthesized response)"

    # Return OpenAI-compatible format
    return {
        "id": f"chatcmpl-mock-{random.randint(1000, 9999)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(prompt.split()) + len(content.split()),
        },
    }
