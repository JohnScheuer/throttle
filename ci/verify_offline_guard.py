"""Prove that CI loaded the Python offline guard before running the suite."""

from __future__ import annotations

import socket
from collections.abc import Callable


def _expect_blocked(label: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except RuntimeError as exc:
        if "offline test blocked" not in str(exc):
            raise AssertionError(f"{label} raised the wrong error: {exc}") from exc
    else:
        raise AssertionError(f"offline guard did not block {label}")


def main() -> None:
    if not getattr(socket, "_throttle_offline_guard_active", False):
        raise SystemExit("offline guard is not active; check PYTHONPATH")

    _expect_blocked(
        "DNS",
        lambda: socket.getaddrinfo("ci-egress-probe.invalid", 443),
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        _expect_blocked(
            "connect",
            lambda: stream.connect(("192.0.2.1", 443)),
        )
        _expect_blocked(
            "connect_ex",
            lambda: stream.connect_ex(("192.0.2.1", 443)),
        )

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as datagram:
        _expect_blocked(
            "sendto",
            lambda: datagram.sendto(b"probe", ("192.0.2.1", 443)),
        )

    # Resolution of a local endpoint is intentionally available to fixture tests.
    socket.getaddrinfo("localhost", 0)
    print("offline guard: non-loopback DNS, connect, connect_ex, and sendto blocked")


if __name__ == "__main__":
    main()
