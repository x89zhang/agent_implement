"""Project defense lifecycle hosted in the benchmark interpreter, not Hermes."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from ..agentdojo_adapter import redact_config_snapshot
from ..config import ToolConfig
from ..middleware import build_middleware_manager, output_revision_limit

GUARDS = (
    "aegis",
    "progent",
    "rope",
    "airguard",
    "clawsentry",
    "janus",
    "stepguard",
    "safeagent",
    "adr",
    "pro2guard",
    "agentspec",
    "llamafirewall",
    "toolsafe",
    "agentdog",
    "agentguard",
)


def replay_guards(cfg, lifecycle_path, directory):
    """Replay one immutable Hermes lifecycle through isolated guard controllers."""
    lifecycle_path, directory = Path(lifecycle_path), Path(directory)
    entries = [
        json.loads(line)
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries or entries[0].get("op") != "initialize":
        raise ValueError("Guard lifecycle must start with initialize")
    sequences = [entry.get("sequence") for entry in entries]
    if sequences != list(range(len(entries))):
        raise ValueError("Guard lifecycle sequence is incomplete or out of order")

    root = directory / "defense_replay"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "replay",
        "source": str(lifecycle_path),
        "event_count": len(entries),
        "enabled": [],
        "trace": [],
        "harness": {},
        "methods": {},
    }
    for name in GUARDS:
        if not getattr(cfg, name).enabled:
            continue
        method_cfg = copy.deepcopy(cfg)
        for other in GUARDS:
            getattr(method_cfg, other).enabled = other == name
        method_dir = root / name
        method_dir.mkdir(exist_ok=False)
        controller = None
        status = {"status": "completed", "directory": str(method_dir)}
        try:
            controller = GuardController(
                method_cfg, entries[0].get("task", ""), method_dir
            )
            for entry in entries:
                controller.dispatch(copy.deepcopy(entry))
        except Exception as exc:
            # A broken detector must not discard the capture or stop its peers.
            status = {
                "status": "failed",
                "directory": str(method_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if controller is not None:
                try:
                    controller.close()
                except Exception as exc:  # cleanup belongs to this detector only
                    status = {
                        "status": "failed",
                        "directory": str(method_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        manifest["enabled"].append(name)
        manifest["methods"][name] = status
        method_output = method_dir / "defenses.json"
        if method_output.exists():
            defense = json.loads(method_output.read_text(encoding="utf-8"))
            manifest["trace"].extend(defense.get("trace", []))
            manifest["harness"].update(defense.get("harness", {}))

    output = directory / "defenses.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=str))
    temporary.replace(output)
    return manifest


class GuardController:
    def __init__(self, cfg, task, directory):
        self.cfg = copy.deepcopy(cfg)
        # The external agent owns planning; only project defenses are installed.
        self.cfg.middleware.enabled = False
        self.cfg.agent.system_prompt = ""
        self.task, self.directory = task, directory
        self.lock = threading.RLock()
        self.manager = None
        self.revisions = 0
        self.state = {
            "messages": [],
            "trace_messages": [],
            "trace": [],
            "trace_stats": {},
            "harness": {},
            "iterations": 0,
            "tool_errors": [],
            "_toolsafe_user_request": task,
            "_agentspec_user_request": task,
            "_progent_user_request": task,
            "_rope_user_request": task,
            "_airguard_user_request": task,
            "_clawsentry_user_request": task,
            "_janus_user_request": task,
            "_stepguard_user_request": task,
            "_safeagent_user_request": task,
            "_adr_user_request": task,
            "_trace_persist": {
                "output_path": str(directory / "defenses.json"),
                "run_dir": str(directory),
            },
        }

    def initialize(self, payload):
        from ..airguard.generator import compile_airguard_authority
        from ..agentguard.scenario import compile_agentguard_scenario
        from ..agentspec.generator import compile_agentspec_rules
        from ..pro2guard.generator import compile_pro2guard_policy
        from ..safeagent.generator import compile_safeagent_rules

        existing = {t.name: t for t in self.cfg.tools}
        self.cfg.tools = [
            existing.get(t["name"])
            or ToolConfig(t["name"], "", t.get("description", ""))
            for t in payload["tools"]
        ]
        self.task = payload.get("task", self.task)
        generation_task = payload.get("generation_task", self.task)
        self.state["_progent_tools"] = copy.deepcopy(payload["tools"])
        self.state["_progent_user_request"] = generation_task
        self.state["_rope_tools"] = copy.deepcopy(payload["tools"])
        self.state["_rope_user_request"] = generation_task
        self.state["_airguard_user_request"] = generation_task
        self.state["_clawsentry_user_request"] = generation_task
        self.state["_janus_user_request"] = generation_task
        self.state["_stepguard_user_request"] = generation_task
        self.state["_stepguard_tools"] = copy.deepcopy(payload["tools"])
        self.state["_safeagent_user_request"] = generation_task
        self.state["_safeagent_tools"] = copy.deepcopy(payload["tools"])
        self.state["_adr_user_request"] = generation_task
        self.state["_toolsafe_user_request"] = self.task
        self.state["_agentspec_user_request"] = self.task
        airguard_generation = compile_airguard_authority(
            self.cfg, generation_task, self.directory
        )
        if airguard_generation.enabled:
            self.state["trace"].append({
                "step": "airguard_authority_generate",
                "output": airguard_generation.to_trace(),
            })
        for compiler in (
            compile_pro2guard_policy,
            compile_agentspec_rules,
            compile_agentguard_scenario,
            compile_safeagent_rules,
        ):
            result = compiler(
                self.cfg, generation_task, self.directory, user_input=""
            )
            if result.enabled:
                self.state["trace"].append(
                    {"step": compiler.__name__, "output": result.to_trace()}
                )
        self.manager = build_middleware_manager(self.cfg)
        return {"enabled": [name for name in GUARDS if getattr(self.cfg, name).enabled]}

    def dispatch(self, payload):
        with self.lock:
            op = payload["op"]
            if op == "initialize":
                if self.manager is not None:
                    raise ValueError("Defense controller already initialized")
                result = self.initialize(payload)
            else:
                if self.manager is None:
                    raise RuntimeError("Defense controller not initialized")
                # `_last_*` values describe one lifecycle operation. Leaving
                # them in shared state makes a decision appear again on every
                # later event, inflating alarm counts and obscuring its phase.
                for key in tuple(self.state):
                    if key.startswith("_last_"):
                        self.state.pop(key, None)
                result = self._dispatch(op, payload)
            current_decisions = {
                key: value
                for key, value in self.state.items()
                if key.startswith("_last_")
            }
            entry = {
                "step": op,
                "timestamp": time.time(),
                "output": result,
                "decisions": current_decisions,
            }
            with (self.directory / "defense_events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(redact_config_snapshot(entry), default=str) + "\n"
                )
            self.state["trace"].append(
                {
                    **entry,
                    "step": "defense_" + op,
                }
            )
            self.persist()
            return result

    def _dispatch(self, op, payload):
        state, manager = self.state, self.manager
        if op == "model_input":
            messages = copy.deepcopy(payload["messages"])
            state["messages"] = messages
            state["trace_messages"] = copy.deepcopy(messages)
            state["iterations"] += 1
            chunks = manager.before_model(state)
            if chunks:
                messages.append({"role": "system", "content": "\n\n".join(chunks)})
            result = asdict(manager.guard_model_input(state, messages))
            result.pop("tool_call", None)
            return result
        if op == "model_output":
            content = payload.get("content") or ""
            calls = payload.get("tool_calls") or []
            accepted = []
            decision = None
            # The project contract is one candidate action per check. Check all parallel calls.
            for call in calls or [None]:
                candidate = (call["name"], call["arguments"]) if call else None
                decision = manager.guard_model_output(state, content, candidate)
                content = decision.content if decision.content is not None else content
                if decision.retry or decision.terminate or not decision.allowed:
                    break
                if decision.tool_call is not None:
                    name, arguments = decision.tool_call
                    accepted.append({**call, "name": name, "arguments": arguments})
            retry = decision.retry and self.revisions < output_revision_limit(self.cfg)
            if retry:
                self.revisions += 1
            elif decision.retry:
                # Never release an unapproved answer after exhausting the revision budget.
                decision.allowed = False
                decision.terminate = True
                content = "Defense revision budget exhausted."
            else:
                self.revisions = 0
            if not decision.allowed or decision.terminate or retry:
                accepted = []
            if not retry:
                manager.after_model(
                    state,
                    content,
                    (accepted[0]["name"], accepted[0]["arguments"])
                    if accepted
                    else None,
                )
            return {
                "allowed": decision.allowed,
                "content": content,
                "tool_calls": accepted,
                "retry": retry,
                "feedback": decision.feedback,
                "reason": decision.reason,
                "terminate": decision.terminate,
            }
        if op == "before_tool":
            return asdict(
                manager.before_tool(state, payload["name"], payload["arguments"])
            )
        if op == "after_tool":
            name, arguments, value = (
                payload["name"],
                payload["arguments"],
                payload["result"],
            )
            decision = manager.after_tool(
                state, name, arguments, value, payload.get("failed", False)
            )
            state["trace"].append(
                {
                    "step": "tool",
                    "tool": name,
                    "input": arguments,
                    "output": decision.result,
                }
            )
            return asdict(decision)
        raise ValueError(f"Unknown defense operation: {op}")

    def persist(self):
        value = {
            "enabled": [n for n in GUARDS if getattr(self.cfg, n).enabled],
            "harness": self.state["harness"],
            "trace": self.state["trace"],
            "trace_stats": self.state["trace_stats"],
            "last_decisions": {
                k: v for k, v in self.state.items() if k.startswith("_last_")
            },
        }
        path = self.directory / "defenses.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(redact_config_snapshot(value), indent=2, default=str)
        )
        temporary.replace(path)

    def close(self):
        from ..agentguard import close_agentguard_session

        cleanup = close_agentguard_session(self.state)
        if cleanup is not None:
            self.state["harness"].setdefault("agentguard", {})["cleanup"] = cleanup
        self.persist()
