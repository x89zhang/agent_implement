"""PIGuard: an outer-circle prompt-injection detector over runtime content.

PIGuard (Li et al., ACL 2025) classifies one text segment as ``benign`` or
``injection``. As a runtime detector it reads content as it enters the agent's
context: the user's request (once) and every tool output, including skill
files read through ``skill_view``. It does not look at actions or history.

The classifier is the authors' released checkpoint run by ``worker.py`` in an
isolated interpreter with upstream's pinned versions; the verdict is the
model's top label, exactly as upstream's pipeline returns it. ``monitor``
records verdicts; ``block`` additionally withholds a flagged tool output (an
integration choice: PIGuard itself only classifies).

Hermes' native skill tools answer with a JSON envelope (``success``, paths,
readiness fields) around the skill. Like the MCP ``{"result": ...}`` envelope
every guard already reads through, it is harness plumbing: the verdict is
taken on the skill text itself (``skill_view``: the SKILL.md body without its
YAML frontmatter; ``skills_list``: the listed descriptions). The raw response
is classified too and recorded as ``raw_*`` for analysis only; PIGuard flags
short JSON snippets regardless of their content.
"""

from __future__ import annotations

import atexit
import json
import re
import selectors
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision
from ..tool_results import unwrap_hermes_result

WORKER = Path(__file__).with_name("worker.py")
INJECTION_LABEL = "injection"
WITHHELD = "Tool output withheld by PIGuard: prompt injection detected."
_LAST_KEY = "_last_piguard_decision"
_FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def skill_text(tool: str, text: str) -> str | None:
    """The skill content inside a Hermes skill-tool response, or None."""
    if tool not in {"skill_view", "skills_list"}:
        return None
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if tool == "skill_view" and isinstance(value.get("content"), str):
        return _FRONTMATTER.sub("", value["content"], count=1).strip()
    if tool == "skills_list" and isinstance(value.get("skills"), list):
        lines = [
            str(skill.get("description") or "").strip()
            for skill in value["skills"] if isinstance(skill, dict)
        ]
        return "\n".join(line for line in lines if line) or str(value.get("message") or "")
    # Failures (e.g. an unknown skill) carry their text in ``error``/``message``.
    for key in ("error", "message"):
        if isinstance(value.get(key), str):
            return value[key]
    return None


class WorkerError(RuntimeError):
    pass


class _Worker:
    """One long-lived classifier process; requests are serialized."""

    def __init__(self, python: str, model_path: str, max_length: int, timeout: float) -> None:
        self.timeout = timeout
        self.lock = threading.Lock()
        self.stderr = tempfile.TemporaryFile(mode="w+")
        try:
            self.process = subprocess.Popen(
                [python, str(WORKER), "--model", model_path, "--max-length", str(max_length)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, bufsize=1,
            )
        except OSError as exc:
            raise WorkerError(f"PIGuard runtime unavailable ({python}): {exc}") from exc
        atexit.register(self.close)
        self.info = self._read()
        if not self.info.get("ready"):
            raise WorkerError(f"PIGuard worker did not start: {self.info}")

    def _read(self) -> dict[str, Any]:
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        if not selector.select(self.timeout):
            raise WorkerError(f"PIGuard worker timed out after {self.timeout:.0f}s")
        line = self.process.stdout.readline()
        if not line:
            self.stderr.seek(0)
            tail = self.stderr.read()[-1500:]
            raise WorkerError(f"PIGuard worker exited (code {self.process.poll()}): {tail}")
        return json.loads(line)

    def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        with self.lock:
            self.process.stdin.write(json.dumps({"texts": texts}) + "\n")
            self.process.stdin.flush()
            answer = self._read()
        if "error" in answer:
            raise WorkerError(f"PIGuard classification failed: {answer['error']}")
        return answer["results"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class PIGuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, classifier: Any = None) -> None:
        self.cfg = cfg
        self.settings = cfg.piguard
        self._classifier = classifier
        self._error = ""

    def _classify(self, texts: list[str]) -> list[dict[str, Any]]:
        if self._classifier is None and not self._error:
            try:
                worker = _Worker(
                    self.settings.python_executable, self.settings.model_path,
                    self.settings.max_length, self.settings.timeout_seconds,
                )
                self._classifier = worker.classify
                self._revision = worker.info.get("revision", "")
            except (WorkerError, OSError, ValueError) as exc:
                self._error = str(exc)
        if self._error:
            raise WorkerError(self._error)
        return self._classifier(texts)

    def _scan(
        self, state: dict[str, Any], phase: str, text: str, tool: str = "",
        raw: str | None = None,
    ) -> dict[str, Any] | None:
        if not text.strip():
            return None
        started = time.monotonic()
        event: dict[str, Any] = {
            "phase": phase, "tool": tool, "mode": self.settings.mode,
            "input_view": "raw" if raw is None else "skill_text",
            "flagged": False, "label": None, "injection_score": None,
            "tokens": None, "truncated": None, "error": "", "source": "piguard",
        }
        try:
            results = self._classify([text] if raw is None else [text, raw])
            result = results[0]
            event.update({
                "label": result["label"],
                "flagged": result["label"] == INJECTION_LABEL,
                "injection_score": result["scores"].get(INJECTION_LABEL),
                "tokens": result.get("tokens"),
                "truncated": result.get("truncated"),
                "revision": getattr(self, "_revision", ""),
            })
            if raw is not None:
                # Diagnostic only: the verdict on the full Hermes response.
                event.update({
                    "raw_label": results[1]["label"],
                    "raw_injection_score": results[1]["scores"].get(INJECTION_LABEL),
                })
        except Exception as exc:  # noqa: BLE001 - recorded as a monitor error
            event.update({"error": str(exc), "source": "error"})
        event["latency_ms"] = round((time.monotonic() - started) * 1000)
        state[_LAST_KEY] = event
        state.setdefault("piguard_events", []).append(event)
        state.setdefault("harness", {})["piguard"] = {
            "enabled": True, "mode": self.settings.mode,
            "status": "error" if event["error"] else "active",
            "event_count": len(state["piguard_events"]), "last_decision": event,
        }
        return event

    def guard_model_input(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> ModelDecision:
        if not self.settings.scan_user_input or state.get("_piguard_user_input_scanned"):
            return ModelDecision()
        state["_piguard_user_input_scanned"] = True
        text = str(state.get("_runtime_user_request") or "")
        if not text:
            text = next(
                (_text(message.get("content")) for message in messages if message.get("role") == "user"),
                "",
            )
        self._scan(state, "user_input", text)
        return ModelDecision()

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        if not self.settings.scan_tool_outputs:
            return ResultDecision(result=result)
        text = unwrap_hermes_result(str(result))
        skill = skill_text(name, text)
        if skill is None:
            event = self._scan(state, "tool_output", text, tool=name)
        else:
            event = self._scan(state, "tool_output", skill, tool=name, raw=text)
        if event is None or self.settings.mode != "block":
            return ResultDecision(result=result)
        if event["flagged"] or (event["error"] and self.settings.fail_closed):
            return ResultDecision(False, event["error"] or "PIGuard flagged prompt injection", WITHHELD, "piguard")
        return ResultDecision(result=result)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""
