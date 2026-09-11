"""Process-wide egress guard: name resolution only for allowlisted hosts.

Every fetch in ``naigos.research`` already calls ``allowlist.check_url`` before
it touches the network. That check covers the code in this repository; it does
not cover what a third-party library does underneath it (``py3dep`` resolves
its own service endpoints). The guard closes that gap for the snapshot build:
it replaces ``socket.getaddrinfo`` so any lookup of a host that is not in the
fixed allowlist raises before a connection can be attempted. ``requests``,
``urllib3``, ``aiohttp`` and ``asyncio`` all resolve through it.

It is installed only in the short-lived subprocess that runs the research
build (``naigos.pipeline.snapshot_build``), never in the parent worker, whose
Modal client must keep talking to Modal. A refusal fails the snapshot, which is
the correct outcome: a source that needs an unlisted host is a source the
allowlist does not cover, and the allowlist is not extended by a scheduler.

What it cannot see: native libraries that resolve and connect in C. GDAL
(via rasterio/rioxarray/py3dep) is the one in this stack, so the build child
also points GDAL's HTTP at a closed port (``snapshot.NATIVE_NETWORK_OFF``).

Modal's own platform-level domain allowlist (``outbound_domain_allowlist``) is
documented for Sandboxes only, and is in beta; this is the in-process
equivalent for a Function. Numeric IP literals are refused too -- the
allowlist names hosts, so a bare address is by definition not on it.
"""

from __future__ import annotations

import socket

from ..research.allowlist import ALLOWLIST

_ORIGINAL_GETADDRINFO = socket.getaddrinfo


class EgressBlocked(OSError):
    """A lookup for a host outside the allowlist."""


def allowlisted_hosts() -> frozenset[str]:
    return frozenset(h.lower() for s in ALLOWLIST.values() for h in s.hosts)


def _host_text(host) -> str:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    return (host or "").strip().rstrip(".").lower()


def install(allowed: frozenset[str] | None = None) -> frozenset[str]:
    """Replace ``socket.getaddrinfo`` for the rest of this process."""
    allowed = frozenset(allowed if allowed is not None else allowlisted_hosts())

    def guarded(host, *args, **kwargs):
        name = _host_text(host)
        if name not in allowed:
            raise EgressBlocked(
                f"egress to {name or '<empty>'!r} refused: not an allowlisted host "
                f"(naigos.research.allowlist). The snapshot build may reach only {sorted(allowed)}."
            )
        return _ORIGINAL_GETADDRINFO(host, *args, **kwargs)

    guarded.__naigos_egress_guard__ = allowed  # type: ignore[attr-defined]
    socket.getaddrinfo = guarded
    return allowed


def uninstall() -> None:
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO


def installed() -> frozenset[str] | None:
    return getattr(socket.getaddrinfo, "__naigos_egress_guard__", None)
