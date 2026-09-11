"""Project defense lifecycle hosted in the benchmark interpreter, not Hermes."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import asdict

from ..agentdojo_adapter import redact_config_snapshot
from ..config import ToolConfig
from ..middleware import build_middleware_manager, output_revision_limit

GUARDS = (
    "aegis",
    "pro2guard",
    "agentspec",
    "llamafirewall",
    "toolsafe",
    "agentdog",
    "agentguard",
)


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
            "_trace_persist": {
                "output_path": str(directory / "defenses.json"),
                "run_dir": str(directory),
            },
        }

    def initialize(self, payload):
        from ..agentguard.scenario import compile_agentguard_scenario
        from ..agentspec.generator import compile_agentspec_rules
        from ..pro2guard.generator import compile_pro2guard_policy

        existing = {t.name: t for t in self.cfg.tools}
        self.cfg.tools = [
            existing.get(t["name"])
            or ToolConfig(t["name"], "", t.get("description", ""))
            for t in payload["tools"]
        ]
        self.task = payload.get("task", self.task)
        self.state["_toolsafe_user_request"] = self.task
        self.state["_agentspec_user_request"] = self.task
        for compiler in (
            compile_pro2guard_policy,
            compile_agentspec_rules,
            compile_agentguard_scenario,
        ):
            result = compiler(self.cfg, self.task, self.directory, user_input="")
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
                result = self._dispatch(op, payload)
            entry = {"step": op, "timestamp": time.time(), "output": result}
            with (self.directory / "defense_events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(redact_config_snapshot(entry), default=str) + "\n"
                )
            self.state["trace"].append(
                {
                    **entry,
                    "step": "defense_" + op,
                    "decisions": {
                        k: v for k, v in self.state.items() if k.startswith("_last_")
                    },
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
