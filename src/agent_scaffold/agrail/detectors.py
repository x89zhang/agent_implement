"""Execute the upstream AGrail detector classes with a local LLM and sandbox."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .upstream_utils import CHAT
from ._vendor.code_tool import CodeDetection
from ._vendor.permission_tool import PermissionDetection
from ._vendor.web_tool import WebDetection

DETECTORS = {
    "OS_environment_detector": CodeDetection,
    "permission_detector": PermissionDetection,
    "html_detector": WebDetection,
}


class BridgeCheckEnvironment:
    """AGrail environment API forwarded to the isolated host runner."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.code: str | None = None

    def put_file(self, content: str, name: str) -> None:
        if name != "code.py" or not isinstance(content, str):
            raise ValueError("AGrail detector produced no checking program")
        self.code = content

    def run_file(self, path: str, user: str = "root") -> Any:
        if path != "/tmp/code.py" or not self.code:
            raise ValueError("AGrail detector requested an unexpected or missing file")
        request = urllib.request.Request(
            self.url + "/run", method="POST",
            data=json.dumps({"code": self.code, "user": user}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "X-AGrail-Token": self.token},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"AGrail detector bridge returned HTTP {exc.code}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("output"), str):
            raise ValueError("AGrail detector bridge returned an invalid response")
        return SimpleNamespace(output=result["output"].encode("utf-8"))

    def close(self) -> None:
        self.code = None


class DockerCheckEnvironment:
    """AGrail put_file/run_file API backed by a disposable networkless container."""

    def __init__(self, image: str, timeout: float = 30.0) -> None:
        self.image = image
        self.timeout = timeout
        self.directory = tempfile.TemporaryDirectory(prefix="agrail-check-")

    def put_file(self, content: str, name: str) -> None:
        if name != "code.py" or not isinstance(content, str):
            raise ValueError("AGrail detector produced no checking program")
        (Path(self.directory.name) / name).write_text(content, encoding="utf-8")

    def run_file(self, path: str, user: str = "root") -> Any:
        if path != "/tmp/code.py":
            raise ValueError("AGrail detector requested an unexpected file")
        process = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges", "--read-only",
             "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
             "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m", "--mount",
             f"type=bind,src={self.directory.name}/code.py,dst=/check.py,readonly",
             "--user", user, "--entrypoint", "python", self.image,
             "/check.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=self.timeout, check=False,
        )
        return SimpleNamespace(output=process.stdout)

    def close(self) -> None:
        self.directory.cleanup()


def run_detector(middleware: Any, state: dict[str, Any], detector: str,
                 name: str, payload: dict[str, Any], checks: list[dict[str, str]]) -> bool:
    if detector not in DETECTORS:
        raise ValueError(f"Unknown AGrail detector: {detector}")
    # Upstream detectors call get_response_from_openai; route those calls to
    # the configured AGrail LLM while leaving the detector algorithms intact.
    calls = 0

    def chat(prompt: str) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls > 6:
            raise RuntimeError("AGrail detector exceeded six LLM calls")
        response = middleware._chat(state, "", prompt)
        return response, 0

    env = state.get("_agrail_agent_environment")
    owned = env is None
    if owned:
        if os.environ.get("AGENT_CONTAINERIZED") == "1":
            url = os.environ.get("AGRAIL_DETECTOR_BRIDGE_URL", "")
            token = os.environ.get("AGRAIL_DETECTOR_BRIDGE_TOKEN", "")
            if not url or not token:
                raise RuntimeError(
                    "AGrail requires an isolated detector environment in the benchmark "
                    "container; no detector bridge is available"
                )
            env = BridgeCheckEnvironment(url, token)
        else:
            env = DockerCheckEnvironment(middleware.cfg.container.image)
    token = CHAT.set(chat)
    try:
        action = middleware._action(state, name, payload)
        if detector == "html_detector" and "target_element" in payload:
            action = (f"Target Element: {payload['target_element']}\n"
                      f"Target Action: {payload.get('target_action', name)}\n"
                      + str(action))
        result, _ = DETECTORS[detector]().get_checking_result(
            agent_environment=env,
            user_information=middleware._request(state),
            agent_action=str(action),
            tool_check_list=checks,
            agent_specification=middleware.settings.agent_specification
                                or middleware.cfg.agent.system_prompt,
            agent_usage_principles=middleware.settings.principles,
            agent_model=middleware.settings.model or middleware.cfg.llm.model,
            user_identity="root",
        )
        if str(result) not in {"True", "False"}:
            raise ValueError("AGrail detector did not return a boolean result")
        return str(result) == "True"
    finally:
        CHAT.reset(token)
        if owned:
            env.close()
