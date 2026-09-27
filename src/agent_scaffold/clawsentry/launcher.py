"""Own a per-run ClawSentry gateway inside an agent container."""

from __future__ import annotations

import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .upstream import CLAWSENTRY_PYTHON, VENV_ENTRY

# Upstream `clawsentry gateway`, started through venv_entry.py for the one
# GPT-5 request-parameter adaptation documented there.
GATEWAY_COMMAND = [CLAWSENTRY_PYTHON, str(VENV_ENTRY), "gateway"]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _gateway_environment(source: dict[str, str], key_env: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
        raise ValueError("Invalid ClawSentry API key environment variable name")
    custom_token = source.get(key_env, "")
    standard_token = source.get("CS_AUTH_TOKEN", "")
    if custom_token and standard_token and custom_token != standard_token:
        raise ValueError("ClawSentry API key environment variables disagree")
    token = custom_token or standard_token or secrets.token_urlsafe(32)
    port = _free_loopback_port()
    env = dict(source)
    env[key_env] = token
    env["CS_AUTH_TOKEN"] = token
    env["CS_HTTP_HOST"] = "127.0.0.1"
    env["CS_HTTP_PORT"] = str(port)
    env["AGENT_CONTAINERIZED"] = "1"
    env["AGENT_CLAWSENTRY_URL"] = f"http://127.0.0.1:{port}"
    # Upstream's benchmark environment (cli/benchmark_command.py:62-74): with
    # the operator bridge on, a pre_action DEFER waits up to 24 h for a human
    # (detection_config.py:139-141). Without it the DEFER is returned at once
    # and recorded; approval-type outcomes are not auto-resolved.
    env["CS_DEFER_BRIDGE_ENABLED"] = "false"
    # Keep local gateway requests off any configured corporate HTTP proxy.
    for name in ("NO_PROXY", "no_proxy"):
        entries = [part.strip() for part in env.get(name, "").split(",") if part.strip()]
        env[name] = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost"]))
    return env


def _config_path(command: list[str]) -> str:
    for index, part in enumerate(command):
        if part == "--config" and index + 1 < len(command):
            return command[index + 1]
        if part.startswith("--config="):
            return part.split("=", 1)[1]
    return ""


def _llm_environment(command: list[str], source: dict[str, str]) -> dict[str, str]:
    """Gateway-only CS_LLM_* settings derived from the agent's run config."""
    path = _config_path(command)
    if not path:
        return {}
    from ..config import load_config
    from .upstream import gateway_llm_environment

    try:
        cfg = load_config(path)
    except Exception as exc:
        raise RuntimeError(f"could not load run config for the gateway LLM: {exc}") from exc
    return gateway_llm_environment(cfg, source)


def _wait_ready(gateway: subprocess.Popen, url: str, token: str, stop: threading.Event, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    request = Request(url + "/health", headers={"Authorization": f"Bearer {token}"})
    while not stop.is_set() and time.monotonic() < deadline:
        if gateway.poll() is not None:
            raise RuntimeError(f"ClawSentry gateway exited before readiness (exit {gateway.returncode})")
        try:
            with urlopen(request, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError, TimeoutError):
            pass
        stop.wait(0.1)
    raise RuntimeError("ClawSentry gateway did not become ready within 30 seconds")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run(command: list[str], *, key_env: str = "CS_AUTH_TOKEN") -> int:
    if not command:
        raise ValueError("Missing agent command")
    env = _gateway_environment(dict(os.environ), key_env)
    # The provider key reaches only the gateway, not the agent process.
    gateway_env = {**env, **_llm_environment(command, env)}
    stop = threading.Event()
    previous = {}

    def handle_signal(signum, _frame):
        stop.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, handle_signal)
    gateway = None
    agent = None
    try:
        gateway = subprocess.Popen(
            GATEWAY_COMMAND, env=gateway_env, start_new_session=True,
        )
        _wait_ready(gateway, env["AGENT_CLAWSENTRY_URL"], env["CS_AUTH_TOKEN"], stop)
        agent = subprocess.Popen(command, env=env, start_new_session=True)
        while agent.poll() is None:
            if stop.is_set():
                return 143
            if gateway.poll() is not None:
                raise RuntimeError(f"ClawSentry gateway exited during agent run (exit {gateway.returncode})")
            stop.wait(0.1)
        return agent.returncode if agent.returncode >= 0 else 128 - agent.returncode
    finally:
        _stop_process(agent)
        _stop_process(gateway)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> int:
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    try:
        return run(command, key_env=os.environ.get("AGENT_CLAWSENTRY_KEY_ENV") or "CS_AUTH_TOKEN")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ClawSentry container startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
