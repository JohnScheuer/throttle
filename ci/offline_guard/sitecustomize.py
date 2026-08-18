"""Fail closed if a Python test tries to use a non-loopback network target."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any


class OfflineNetworkBlocked(RuntimeError):
    """Raised before a test can resolve or contact a non-loopback target."""


_real_getaddrinfo = socket.getaddrinfo
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto


def _is_loopback(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    if not isinstance(value, str):
        return False
    host = value.strip("[]").rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_from_address(address: object) -> object:
    if isinstance(address, tuple) and address:
        return address[0]
    return None


def _guard_address(sock: socket.socket, address: object, operation: str) -> None:
    if sock.family == socket.AF_UNIX:
        return
    host = _host_from_address(address)
    if not _is_loopback(host):
        raise OfflineNetworkBlocked(
            f"offline test blocked non-loopback {operation}: {host!r}"
        )


def _offline_getaddrinfo(
    host: object, *args: object, **kwargs: object
) -> list[tuple[Any, ...]]:
    if not _is_loopback(host):
        raise OfflineNetworkBlocked(
            f"offline test blocked non-loopback DNS resolution: {host!r}"
        )
    return _real_getaddrinfo(host, *args, **kwargs)


def _offline_connect(
    sock: socket.socket, address: object
) -> None:
    _guard_address(sock, address, "connect")
    return _real_connect(sock, address)  # type: ignore[arg-type]


def _offline_connect_ex(sock: socket.socket, address: object) -> int:
    _guard_address(sock, address, "connect_ex")
    return _real_connect_ex(sock, address)  # type: ignore[arg-type]


def _offline_sendto(sock: socket.socket, data: bytes, *args: object) -> int:
    if not args:
        raise TypeError("sendto requires a destination")
    address = args[-1]
    _guard_address(sock, address, "sendto")
    return _real_sendto(sock, data, *args)  # type: ignore[arg-type]


socket.getaddrinfo = _offline_getaddrinfo  # type: ignore[assignment]
socket.socket.connect = _offline_connect  # type: ignore[assignment]
socket.socket.connect_ex = _offline_connect_ex  # type: ignore[assignment]
socket.socket.sendto = _offline_sendto  # type: ignore[assignment]
socket._throttle_offline_guard_active = True  # type: ignore[attr-defined]
