"""Own a per-run upstream AgentGuard server (``src/server/backend``).

Upstream deploys the policy decision point as a separate server
(``docker-compose.yml``: ``AGENTGUARD_POLICY``, ``AGENTGUARD_SERVER_PLUGIN_CONFIG``
and ``AGENTGUARD_LLM_*``) and points the client at it with ``server_url``. The
client alone never evaluates a policy (``u_guard/enforcer.py`` returns
``local_no_remote`` allow), so every run gets its own loopback server loaded with
that run's policy file.

``python -m agent_scaffold.agentguard.launcher serve`` is the server process.
It runs upstream's ``uvicorn backend.api.app:app`` (``scripts/entrypoint.sh``)
when FastAPI and uvicorn are installed, and otherwise upstream's stdlib server
(``backend/api/dev_server.py``) bound to the same ``app_state`` singletons.
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor"
DEFAULT_PLUGIN_CONFIG = VENDOR_ROOT / "config" / "plugins.json"
_LLM_ENV = "AGENT_SCAFFOLD_AGENTGUARD_LLM"


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass
class AgentGuardServer:
    url: str
    api_key: str
    process: subprocess.Popen
    log_path: str = ""
    _log: Any = field(default=None, repr=False)

    def running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        process = self.process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if self._log is not None:
            self._log.close()
            self._log = None


def _server_environment(
    *,
    port: int,
    api_key: str,
    policy_path: str,
    plugin_config: str,
    llm: dict[str, Any] | None,
) -> dict[str, str]:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    paths = [str(VENDOR_ROOT / "server"), str(VENDOR_ROOT), source_root]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["AGENTGUARD_HOST"] = "127.0.0.1"
    env["AGENTGUARD_PORT"] = str(port)
    env["AGENTGUARD_API_KEY"] = api_key
    env["AGENTGUARD_POLICY"] = policy_path or ""
    env["AGENTGUARD_SERVER_PLUGIN_CONFIG"] = plugin_config or ""
    llm = dict(llm or {})
    api_key_value = str(llm.pop("api_key", "") or "")
    if api_key_value:
        env["AGENTGUARD_LLM_API_KEY"] = api_key_value
    if llm.get("base_url"):
        env["AGENTGUARD_LLM_BASE_URL"] = str(llm["base_url"])
    if llm.get("model"):
        env["AGENTGUARD_LLM_MODEL"] = str(llm["model"])
    env[_LLM_ENV] = json.dumps(llm)
    # Keep loopback requests off any configured HTTP proxy.
    for name in ("NO_PROXY", "no_proxy"):
        entries = [part.strip() for part in env.get(name, "").split(",") if part.strip()]
        env[name] = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost"]))
    return env


def _wait_ready(
    process: subprocess.Popen, url: str, api_key: str, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    # Same probe as upstream docker-compose's healthcheck.
    request = Request(url + "/v1/backend/health", headers={"X-Api-Key": api_key})
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"AgentGuard server exited before readiness (exit {process.returncode})"
            )
        try:
            with urlopen(request, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"AgentGuard server did not become ready within {timeout:g} seconds")


def start_server(
    *,
    policy_path: str,
    plugin_config: str = "",
    llm: dict[str, Any] | None = None,
    work_dir: str = "",
    timeout: float = 30.0,
) -> AgentGuardServer:
    """Start one upstream AgentGuard server on a free loopback port.

    ``work_dir`` (the run directory) receives ``agentguard_server.log`` and any
    files the server writes relative to its working directory.
    """

    directory = Path(work_dir) if work_dir else Path.cwd()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = str(directory / "agentguard_server.log")

    port = _free_loopback_port()
    api_key = secrets.token_urlsafe(24)
    env = _server_environment(
        port=port,
        api_key=api_key,
        policy_path=policy_path,
        plugin_config=plugin_config or str(DEFAULT_PLUGIN_CONFIG),
        llm=llm,
    )
    log = open(log_path, "ab")  # noqa: SIM115
    process = subprocess.Popen(
        [sys.executable, "-m", "agent_scaffold.agentguard.launcher", "serve"],
        env=env,
        cwd=str(directory),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    server = AgentGuardServer(
        url=f"http://127.0.0.1:{port}",
        api_key=api_key,
        process=process,
        log_path=log_path,
        _log=log,
    )
    atexit.register(server.stop)
    try:
        _wait_ready(process, server.url, api_key, timeout)
    except Exception:
        server.stop()
        raise
    return server


# ---- server process ---------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


class _ScaffoldProvider:
    """Server LLM provider backed by the project's LLM adapter.

    Upstream's ``OpenAICompatibleProvider`` posts to ``/chat/completions`` with
    ``temperature`` and ``max_tokens``; the models used here are Responses-API
    models, so the same single-prompt completion is sent through the adapter.
    """

    name = "agent_scaffold"

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def complete(self, prompt: str, **_: Any) -> str:
        return _text(self._adapter.chat([{"role": "user", "content": prompt}]).content)


def _install_llm_provider() -> None:
    raw = os.environ.get(_LLM_ENV, "")
    settings = json.loads(raw) if raw else {}
    if not settings.get("model"):
        return
    from agent_scaffold.config import LLMConfig
    from agent_scaffold.llm import LLMAdapter

    import backend.llm.llm_client as llm_client

    config = LLMConfig(
        provider=str(settings.get("provider") or "openai"),
        model=str(settings["model"]),
        temperature=float(settings.get("temperature") or 0.0),
        base_url=str(settings.get("base_url") or ""),
        api_key=os.environ.get("AGENTGUARD_LLM_API_KEY", ""),
        api_key_env=str(settings.get("api_key_env") or ""),
        request_timeout=settings.get("request_timeout"),
    )
    upstream_get_provider = llm_client.get_provider

    def get_provider(**kwargs: Any) -> Any:
        config_override = dict(kwargs.get("config") or {})
        if str(config_override.get("backend") or "").lower() in {"heuristic", "offline"}:
            return upstream_get_provider(**kwargs)
        return _ScaffoldProvider(LLMAdapter(config))

    llm_client.get_provider = get_provider


def serve() -> int:
    for path in (str(VENDOR_ROOT), str(VENDOR_ROOT / "server")):
        if path not in sys.path:
            sys.path.insert(0, path)
    _install_llm_provider()
    host = os.environ.get("AGENTGUARD_HOST", "127.0.0.1")
    port = int(os.environ.get("AGENTGUARD_PORT", "38080"))
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        uvicorn = None
    if uvicorn is not None:
        uvicorn.run("backend.api.app:app", host=host, port=port, log_level="warning")
        return 0

    from backend.api.dev_server import start_dev_server
    from backend.app_state import get_console, get_manager, get_skills

    _, server, thread = start_dev_server(
        port, manager=get_manager(), console=get_console(), skills=get_skills()
    )
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    while not stop.is_set() and thread.is_alive():
        stop.wait(0.5)
    server.shutdown()
    get_manager().stop_session_health_monitor()
    return 0


def main() -> int:
    command = sys.argv[1:] or ["serve"]
    if command[0] != "serve":
        print(f"unsupported command: {command[0]}", file=sys.stderr)
        return 2
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
