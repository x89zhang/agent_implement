"""Explicit benchmark session ownership and a private, serialized tool bridge.

The MCP proxy can restart without resetting this session. Evaluation is only a
Python controller operation and is never present on the network tool surface.
"""

from __future__ import annotations

import hmac
import importlib
import json
import secrets
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ADAPTERS = {
    "agentdojo": ("agentdojo_adapter", "AgentDojoSession"),
    "agentharm": ("agentharm_adapter", "AgentHarmSession"),
    "agent_security_bench": (
        "agent_security_bench_adapter",
        "AgentSecurityBenchSession",
    ),
}


def create_benchmark(cfg):
    selected = [name for name in ADAPTERS if getattr(cfg, name).enabled]
    if len(selected) != 1:
        raise ValueError("Hermes requires exactly one supported benchmark")
    name = selected[0]
    module_name, class_name = ADAPTERS[name]
    module = importlib.import_module(f"agent_scaffold.{module_name}")
    config = getattr(cfg, name)
    module.build_tool_configs(config)
    session = getattr(module, class_name)(config)
    tools = []
    if name == "agentdojo":
        definitions = session.suite.tools
    elif name == "agentharm":
        definitions = session.tools.values()
    else:
        definitions = []
        for tool_name, row in module._TOOL_METADATA.items():
            tools.append(
                {
                    "name": tool_name,
                    "description": str(row.get("Description") or tool_name),
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            )
    for tool in definitions:
        parameters = tool.parameters
        # AgentDojo has a parameter model class; Inspect ToolDef exposes a schema model instance.
        if isinstance(parameters, type):
            schema = parameters.model_json_schema()
        else:
            schema = parameters.model_dump(exclude_none=True, by_alias=True)
        tools.append(
            {"name": tool.name, "description": tool.description, "inputSchema": schema}
        )
    prompt_config = (
        replace(config, injection_method="clean")
        if name == "agent_security_bench" and cfg.execution.memory.mode != "off"
        else config
    )
    task = module.augment_task(cfg.agent.task.strip(), prompt_config)
    return name, session, tools, task


class BenchmarkBridge:
    def __init__(self, session, tools: list[dict], journal: Path, guard=None):
        self.session, self.tools, self.journal = session, tools, journal
        self.guard = guard
        self.guard_token = secrets.token_urlsafe(32)
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._active = True
        self.calls: list[dict] = []
        self._names = {tool["name"] for tool in tools}
        self._schemas = {tool["name"]: tool["inputSchema"] for tool in tools}
        self._server = None
        self._thread = None

    def execute(self, name, arguments):
        with self._lock:
            if not self._active:
                raise RuntimeError("Benchmark phase has ended")
            if name not in self._names or not isinstance(arguments, dict):
                raise ValueError("Unknown tool or invalid arguments")
            import jsonschema

            try:
                jsonschema.validate(arguments, self._schemas[name])
            except jsonschema.ValidationError as exc:
                return {
                    "text": f"Invalid tool arguments: {exc.message}",
                    "is_error": True,
                }
            event = {
                "step": "tool",
                "timestamp": time.time(),
                "tool": name,
                "arguments": arguments,
                "sequence": len(self.calls),
                "status": "started",
            }
            self.calls.append(event)
            self._append(event)
            try:
                text = self.session.run_tool(name, arguments)
                result = {"text": str(text), "is_error": False}
                event = {**event, "status": "completed", "result": str(text)}
            except Exception as exc:  # noqa: BLE001 - serialize failures at the process/tool boundary
                result = {"text": str(exc), "is_error": True}
                event = {**event, "status": "failed", "error": str(exc)}
            self.calls[-1] = event
            self._append(event)
            return result

    def _append(self, value):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def __enter__(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""),
                    "Bearer "
                    + (bridge.guard_token if self.path == "/guard" else bridge.token),
                ):
                    self.send_error(403)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size < 0 or size > 8 * 1024 * 1024:
                        raise ValueError("Request exceeds bridge size limit")
                    payload = json.loads(self.rfile.read(size))
                    if self.path == "/guard" and bridge.guard is not None:
                        result = bridge.guard.dispatch(payload)
                    elif self.path == "/tools":
                        result = bridge.tools
                    elif self.path == "/call":
                        result = bridge.execute(
                            payload["name"], payload.get("arguments", {})
                        )
                    else:
                        self.send_error(404)
                        return
                    data = json.dumps(result, ensure_ascii=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (ValueError, KeyError, RuntimeError) as exc:
                    self.send_error(400, str(exc))

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        # Worker shutdown happens before this barrier. Never evaluate a mutating session concurrently.
        self._active = False

    def evaluate(self, final_output):
        with self._lock:
            return self.session.evaluate(final_output)


def _service_process(cfg, connection, journal):
    """Keep benchmark code and potentially blocking tools in a killable process."""
    import contextlib
    import os

    os.setsid()
    with (
        journal.with_name("service.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            name, session, tools, task = create_benchmark(cfg)
            from .guards import GuardController

            guard = GuardController(cfg, task, journal.parent)
            with BenchmarkBridge(session, tools, journal, guard) as bridge:
                connection.send(
                    {
                        "name": name,
                        "tools": tools,
                        "task": task,
                        "url": bridge.url,
                        "token": bridge.token,
                        "guard_token": bridge.guard_token,
                    }
                )
                while True:
                    command = connection.recv()
                    if command["op"] == "evaluate":
                        connection.send(
                            {
                                "evaluation": bridge.evaluate(command["final_output"]),
                                "calls": bridge.calls,
                            }
                        )
                    elif command["op"] == "close":
                        guard.close()
                        break
                    else:
                        raise ValueError("Unknown controller operation")
        except EOFError:
            pass
        except Exception as exc:  # noqa: BLE001 - serialize failures at the process/tool boundary
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            connection.close()


class BenchmarkService:
    """Private pipe for lifecycle/evaluation; only tool operations reach MCP."""

    def __init__(self, cfg, journal: Path, timeout: float):
        import multiprocessing

        self._context = multiprocessing.get_context("spawn")
        self._parent, self._child = self._context.Pipe()
        self._process = self._context.Process(
            target=_service_process, args=(cfg, self._child, journal)
        )
        self.timeout = timeout
        self.calls = []

    def _receive(self):
        if not self._parent.poll(self.timeout):
            raise TimeoutError(
                "Benchmark service exceeded its startup/evaluation deadline"
            )
        try:
            result = self._parent.recv()
        except EOFError:
            raise RuntimeError("Benchmark service exited unexpectedly") from None
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def __enter__(self):
        self._process.start()
        self._child.close()
        try:
            ready = self._receive()
            self.name, self.tools, self.task = (
                ready["name"],
                ready["tools"],
                ready["task"],
            )
            self.url, self.token = ready["url"], ready["token"]
            self.guard_token = ready["guard_token"]
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def evaluate(self, final_output):
        self._parent.send({"op": "evaluate", "final_output": final_output})
        result = self._receive()
        self.calls = result["calls"]
        return result["evaluation"]

    def __exit__(self, *args):
        import os
        import signal

        try:
            if self._process.is_alive():
                try:
                    self._parent.send({"op": "close"})
                except (BrokenPipeError, OSError):
                    pass
                self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
            # Also stop any subprocesses started by benchmark tool code.
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self._process.is_alive():
                self._process.kill()
            self._process.join(timeout=3)
        finally:
            self._parent.close()
            self._process.close()
