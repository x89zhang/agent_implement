"""Host-side AGrail check runner for containerized benchmark replay.

Generated checking programs execute only in a short-lived Docker container
with networking disabled and a read-only root filesystem.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .detectors import DockerCheckEnvironment


class DetectorBridge:
    def __init__(self, image: str) -> None:
        self.image = image
        self.token = secrets.token_urlsafe(32)
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if self.path != "/run" or not hmac.compare_digest(
                    self.headers.get("X-AGrail-Token", ""), bridge.token
                ):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(400)
                    return
                if not 0 < length <= 200_000:
                    self.send_error(413)
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                    code = payload["code"]
                    user = payload.get("user", "root")
                    if not isinstance(code, str) or not code:
                        raise ValueError("code must be a nonempty string")
                    if not isinstance(user, str) or not user:
                        raise ValueError("user must be a nonempty string")
                    environment = DockerCheckEnvironment(bridge.image)
                    try:
                        environment.put_file(code, "code.py")
                        output = environment.run_file("/tmp/code.py", user=user).output
                    finally:
                        environment.close()
                    result = {"output": output[:100_000].decode("utf-8", errors="replace")}
                    status = 200
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    status = 500
                body = json.dumps(result).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
