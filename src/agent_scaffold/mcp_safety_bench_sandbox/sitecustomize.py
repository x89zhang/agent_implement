"""Loaded by the MCP-SafetyBench servers' interpreter (on their PYTHONPATH only).

Makes the hosts named in the task's attack text unresolvable, so a poisoned
call (e.g. fetch https://attacker.com/collect) never reaches a real third
party. Grading reads the call's arguments, not its network effect.
"""

import os
import socket

_BLOCKED = {
    host.strip().lower().rstrip(".")
    for host in os.environ.get("MCP_SAFETY_BENCH_BLOCKED_HOSTS", "").split(",")
    if host.strip()
}

if _BLOCKED:
    _getaddrinfo = socket.getaddrinfo

    def _blocked(host) -> bool:
        if isinstance(host, bytes):
            host = host.decode(errors="ignore")
        name = str(host or "").lower().rstrip(".")
        return any(name == item or name.endswith("." + item) for item in _BLOCKED)

    def getaddrinfo(host, *args, **kwargs):
        if _blocked(host):
            raise socket.gaierror(socket.EAI_NONAME, "blocked by the benchmark sandbox")
        return _getaddrinfo(host, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo
