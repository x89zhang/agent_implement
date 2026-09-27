"""DRIFT's three defenses on the project's shared guard lifecycle.

The upstream implementation lives inside an AgentDojo runner (DRIFTLLM.py) and
an ASB agent (ASB_DRIFT/pyopenagi/agents/drift.py). This adapter keeps their
task-derived trajectory/checklist, tool-result injection isolation, and
pre-execution validation while using the project's own agent and model.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..llm import LLMAdapter
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from ..progent.tools import tool_definitions_from_config
from .upstream_alignment import GUIDELINES as _ALIGNMENT_SYSTEM
from .policy import (
    UPSTREAM_REVISION,
    injection_isolate,
    insert_checklist_step,
    node_check,
    parse_constraints,
    parse_detected_instructions,
    remove_sentence,
    repair_json,
)
from .prompts import (
    ASB_ALIGNMENT_PROMPT,
    ASB_INJECTION_DETECTION_PROMPT,
    CHECKLIST_FORMAT_PROMPT,
    CONSTRAINTS_BUILD_PROMPT,
    ENVIRONMENT_GUIDELINES,
    EXECUTION_GUIDELINES_PROMPT,
    INJECTION_DETECTION_PROMPT,
    PRIVILEGE_PROMPT,
)


class DriftMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, llm: Any | None = None,
                 approval_callback: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.drift
        self._llm = llm
        self.approval_callback = approval_callback
        if self.settings.request_user_approval and approval_callback is None:
            raise ValueError("DRIFT user approval requires an approval callback")
        profile = self.settings.profile
        if profile == "auto":
            asb = getattr(cfg, "agent_security_bench", None)
            profile = "asb" if asb is not None and asb.enabled else "agentdojo"
        self.profile = profile

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        # Each model call is one upstream query(): discard the unfinished
        # validation of an output that was sent back for revision.
        state.pop("_drift_pending", None)
        state.pop("_drift_formatted", None)
        error = self._ensure_constraints(state)
        if error and self.settings.fail_closed:
            self._record(
                state,
                "model_input",
                allowed=False,
                enforced=True,
                reason=error,
                source="error",
            )
            return ModelDecision(
                False,
                error,
                messages=messages,
                content="DRIFT could not initialize its policy.",
                terminate=True,
            )
        return ModelDecision(messages=messages)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        chunks: list[str] = []
        warning = state.pop("_drift_warning", "")
        if warning:
            chunks.append(f"DRIFT warning: {warning}")
        # Upstream appends the execution guidelines to the agent's system
        # prompt on every call once a plan exists (client.py:59-61,
        # DRIFTLLM.py:646). In replay the recorded agent never sees this.
        if (
            self.profile == "agentdojo"
            and self.settings.dynamic_validation
            and not self._ensure_constraints(state)
            and state.get("_drift_trajectory")
        ):
            chunks.append(
                EXECUTION_GUIDELINES_PROMPT.format(
                    initial_trajectory=state["_drift_trajectory"],
                    node_checklist=state.get("_drift_checklist", "None"),
                    achieved_trajectory=state.get("_drift_completed") or [],
                    query=self._task(state),
                )
            )
        return chunks

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        if not tool_call or not self.settings.dynamic_validation:
            return ModelDecision(content=content, tool_call=tool_call)
        error = self._ensure_constraints(state)
        if error:
            if self.settings.fail_closed:
                return ModelDecision(
                    False, error, content=content, tool_call=None, terminate=True
                )
            return ModelDecision(content=content, tool_call=tool_call)
        name, arguments = tool_call
        if self.profile == "asb":
            self._validate_asb(state, name, arguments)
            return ModelDecision(content=content, tool_call=tool_call)
        reason, feedback, source = self._validate(state, name, arguments)
        if not reason:
            return ModelDecision(content=content, tool_call=tool_call)
        enforced = self.settings.mode == "block"
        self._record(
            state,
            "model_output",
            allowed=False,
            enforced=enforced,
            reason=reason,
            source=source,
            tool=name,
        )
        if not enforced:
            if self.settings.mode == "warn":
                state["_drift_warning"] = reason
            return ModelDecision(content=content, tool_call=tool_call)
        return ModelDecision(
            False,
            reason,
            content=content,
            tool_call=None,
            retry=True,
            # Upstream wraps the [CALL ERROR] message dict this way (DRIFTLLM.py:686, 691).
            feedback=f"</function_error>\n{ {'role': 'user', 'content': feedback} }\n</function_error>",
            decision_type="drift_constraint",
        )

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        # The output was released: its calls join the achieved trajectory, as
        # at the end of trajectory_constraint_validation (DRIFTLLM.py:522).
        pending = state.pop("_drift_pending", None)
        if pending is not None:
            state["_drift_completed"] = list(pending)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        if not self.settings.dynamic_validation:
            return ToolDecision()
        error = self._ensure_constraints(state)
        if error:
            enforced = self.settings.fail_closed
            self._record(
                state, "before_tool", allowed=False, enforced=enforced,
                reason=error, source="error", tool=name,
            )
            return ToolDecision(
                not enforced, error if enforced else "",
                decision_type="drift" if enforced else "",
            )
        # Upstream validates at model output only; the ASB profile drops the
        # refused calls of a step before they execute (react_agent_attack.py:312-341).
        denied = state.get("_drift_asb_denied") or []
        key = [name, payload]
        if key not in denied:
            return ToolDecision()
        denied.remove(key)
        reason = f"DRIFT trajectory deviation {name} was refused"
        return ToolDecision(False, reason, decision_type="drift")

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        # Upstream isolates every latest tool message, including errors.
        if not self.settings.injection_isolation:
            return ResultDecision(result=result)
        started = time.monotonic()
        text = str(result)
        error = ""
        detections = 0
        try:
            if self.profile == "asb":
                text, detections = self._isolate_asb(state, name, payload, text)
            else:
                text, detections = self._isolate(state, name, payload, text, failed)
        except Exception as exc:
            error = f"DRIFT isolation failed: {type(exc).__name__}: {exc}"
        masked = not error and text != str(result)
        enforced = bool(error) and self.settings.fail_closed
        event = self._record(
            state,
            "after_tool",
            allowed=not (masked or error),
            enforced=enforced,
            reason=error or ("masked conflicting instructions" if masked else ""),
            source="error" if error else "injection_isolation",
            tool=name,
            detail={
                "masked": masked,
                "detections": detections,
                "latency_ms": round((time.monotonic() - started) * 1000),
            },
        )
        if enforced:
            # Only a failed isolation-model call withholds; upstream would stop the run.
            return ResultDecision(
                False, error, "Tool result withheld by DRIFT.", "drift_isolation_error"
            )
        if error or self.settings.mode == "monitor":
            return ResultDecision(result=result)
        if masked and self.settings.mode == "warn":
            state["_drift_warning"] = event["reason"]
        return ResultDecision(result=text)

    def _isolate(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        text: str,
        failed: bool,
    ) -> tuple[str, int]:
        # DRIFTLLM.py:615-635. Upstream passes the tool message dict's str().
        query = self._task(state)

        def observation(content: str) -> str:
            return str({
                "role": "tool",
                "content": content,
                "tool_call": {"function": name, "args": payload},
                "error": content if failed else None,
            })

        user_prompt = f"""<User Query>\n{query}\n</User Query>
                <Tool Results>\n{observation(text)}\n</Tool Results>"""
        detected = self._complete(state, INJECTION_DETECTION_PROMPT, user_prompt)
        completed, text = injection_isolate(detected, text)
        detections = 1
        cycle_times = 0
        while not completed and cycle_times < self.settings.mask_limitation:
            cycle_times += 1
            user_prompt = f"""<User Query>\n{query}\n</User Query>
                    <Tool Results>\n{observation(text)}\n</Tool Results>"""
            detected = self._complete(state, INJECTION_DETECTION_PROMPT, user_prompt)
            completed, text = injection_isolate(detected, text)
            detections += 1
        return text, detections

    def _isolate_asb(
        self, state: dict[str, Any], name: str, payload: dict[str, Any], text: str
    ) -> tuple[str, int]:
        # ASB_DRIFT/pyopenagi/agents/drift.py:143-206 with cycle_limit=1;
        # action/observation strings follow react_agent_attack.py:145-146.
        conversations = [
            message for message in state.get("messages") or []
            if message.get("role") != "system"
        ]
        tool_call = f"[Action]: I will call the {name} with the params as {payload}"
        observations = [f"The knowledge I get from {name} is: {text}"]
        data = f"""
                        <User Query>\n{self._task(state)}\n</User Query>
                        <Interaction History>\n{conversations}\n</Interaction History>
                        <Tool Call>\n{tool_call}\n</Tool Call>
                        <Observation>\n{observations}\n</Observation>
                        """
        response = self._complete(state, ASB_INJECTION_DETECTION_PROMPT, data)
        for item in parse_detected_instructions(response) or []:
            text = remove_sentence(text, item)
        return text, 1

    def _ensure_constraints(self, state: dict[str, Any]) -> str:
        if state.get("_drift_initialized"):
            return str(state.get("_drift_init_error") or "")
        state["_drift_initialized"] = True
        if not self.settings.dynamic_validation:
            return ""
        started = time.monotonic()
        error = ""
        completion = ""
        # agent_run (client.py:45-61): tools and AgentDojo's environment setup
        # are appended to CONSTRAINTS_BUILD_PROMPT; the query is the user turn.
        # The ASB profile has no generated plan upstream (its trajectory is the
        # ASB agent's own workflow, which Hermes does not expose), so the same
        # generator stands in for it.
        system = (
            CONSTRAINTS_BUILD_PROMPT
            + f"\n\n<avaliable_tools>\n\n{json.dumps(self._tools_docs(state))}\n\n</avaliable_tools>"
        )
        if self.profile == "agentdojo" and self.cfg.agentdojo.enabled:
            system += f"\n\n<environment_setup>\n\n{ENVIRONMENT_GUIDELINES}\n\n</environment_setup>"
        try:
            completion = self._complete(state, system, self._task(state))
        except Exception as exc:
            error = f"DRIFT constraint generation failed: {type(exc).__name__}: {exc}"
        trajectory, checklist = parse_constraints(completion)
        state["_drift_trajectory"] = trajectory
        state["_drift_initial_trajectory"] = list(trajectory)
        state["_drift_checklist"] = checklist
        state["_drift_completed"] = []
        state["_drift_init_error"] = error
        event = {
            "step": "drift_constraints_generate",
            "timestamp": time.time(),
            "output": {
                "source": "llm",
                "profile": self.profile,
                "status": "failed" if error else "ready",
                "trajectory": trajectory,
                "parameter_checklist": checklist,
                "error": error,
                "upstream_revision": UPSTREAM_REVISION,
            },
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
        state.setdefault("trace", []).append(event)
        self._artifact(state, "drift_constraints.json", event)
        return error

    def _validate(
        self, state: dict[str, Any], name: str, arguments: dict[str, Any]
    ) -> tuple[str, str, str]:
        """One call of trajectory_constraint_validation + checklist validation.

        Hermes checks parallel calls one at a time, so ``_drift_pending`` holds
        upstream's temporary achieved trajectory across the calls of one model
        output (DRIFTLLM.py:445-449).
        """
        query = self._task(state)
        pending = state.get("_drift_pending")
        if pending is None:
            pending = list(state.get("_drift_completed") or [])
            state["_drift_pending"] = pending
        trajectory = state.get("_drift_trajectory") or []
        position = len(pending)
        if not (position < len(trajectory) and trajectory[position] == name):
            extended = list(trajectory)
            extended.insert(position, name)
            reason = (
                f"The function trajectory of {extended} is not aligned with "
                f"trajectory plan of {trajectory}."
            )
            aligned = self._deviation_aligned(state, name, trajectory, extended)
            if not aligned and self.settings.request_user_approval:
                aligned = bool(self.approval_callback({
                    "tool": name, "trusted_user_task": query,
                    "initial_function_trajectory": trajectory,
                    "current_function_trajectory": extended,
                }))
            if not aligned:
                achieved = state.get("_drift_completed") or []
                return reason, (
                    f"[CALL ERROR] The function calling of {name} has been refused due to it does not align with original planned trajectory, please change to call other functions to complete original user query.\n"
                    "Remember, do not stop working on the original user task to do other things.\n"
                    f"Below is the specific Refusal Reason:\n {reason}.\n"
                    f"Original Planned Trajecotry:\n{trajectory}.\n"
                    f"Achieved Function Trajectory:\n{achieved}.\n"
                    f"User Query:\n{query}"
                ), "constraint"
            # Accepted deviations extend the plan at validation time (DRIFTLLM.py:492-499).
            state["_drift_trajectory"] = extended
            state["_drift_checklist"] = insert_checklist_step(
                str(state.get("_drift_checklist", "None")), position, name
            )
            pending.append(name)
            state["_drift_completed"] = list(pending)
            self._record(
                state,
                "trajectory_progress",
                allowed=True,
                enforced=False,
                reason="",
                source="dynamic_alignment",
                tool=name,
                detail={"trajectory": list(extended)},
            )
        else:
            pending.append(name)
        if not state.get("_drift_formatted"):
            # node_json_formatting runs before every checklist check (DRIFTLLM.py:538).
            state["_drift_formatted"] = True
            state["_drift_checklist"] = self._format_checklist(
                state, str(state.get("_drift_checklist", "None"))
            )
        checklist = str(state.get("_drift_checklist", "None"))
        passed, reason = node_check(checklist, len(pending), [(name, arguments)])
        if passed:
            return "", "", ""
        # Upstream already recorded this output's calls as achieved (DRIFTLLM.py:522).
        state["_drift_completed"] = list(pending)
        call = {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return reason, (
            f"[CALL ERROR] The function calling of {[call]} has been refused due to some parameters are not aligned with checklist, Please try an alternative method to continue fulfilling the original user query. Remember, do not stop working on the original user task to do other things.\n"
            f"The specific error:\n{reason}\n"
            f"Checklist:\n{checklist}.\n"
            "Note: the value of None denotes this value is uncertain.\n"
            f"User Query:\n{query}"
        ), "checklist"

    def _deviation_aligned(
        self,
        state: dict[str, Any],
        name: str,
        trajectory: list[str],
        extended: list[str],
    ) -> bool:
        # DRIFTLLM.py:479-489: Read tools extend the plan; others ask the
        # alignment judge. Any exception, including an unknown tool, accepts.
        answer = ""
        try:
            permission = self._permission(state, name)
            if permission == "Read":
                aligned = True
            else:
                data = f"""
                <Initial_Function_Trajectory>\n{trajectory}\n</Initial_Function_Trajectory>
                <Current_Function_Trajectory>\n{extended}\n</Current_Function_Trajectory>
                <User_Query>\n{self._task(state)}\n</User_Query>
                """
                answer = self._complete(state, _ALIGNMENT_SYSTEM, data)
                aligned = ("Yes" in answer) or ("YES" in answer)
        except Exception as exc:
            answer = f"{type(exc).__name__}: {exc}"
            aligned = True
        self._record(
            state,
            "trajectory_alignment",
            allowed=aligned,
            enforced=not aligned and self.settings.mode == "block",
            reason="" if aligned else answer,
            source="dynamic_alignment",
            tool=name,
            detail={"proposed_trajectory": extended},
        )
        return aligned

    def _validate_asb(
        self, state: dict[str, Any], name: str, arguments: dict[str, Any]
    ) -> None:
        # react_agent_attack.py:312-341: in-plan calls advance; the judge is
        # consulted but its result is overwritten with False, and privileges
        # are never used, so every deviation is refused.
        achieved = state.setdefault("_drift_completed", [])
        trajectory = state.get("_drift_trajectory") or []
        position = len(achieved)
        if position < len(trajectory) and trajectory[position] == name:
            achieved.append(name)
            return
        data = f"""
                <Initial_Function_Trajectory>\n{trajectory}\n</Initial_Function_Trajectory>
                <Current_Function_Trajectory>\n{[*achieved, name]}\n</Current_Function_Trajectory>
                <User_Query>\n{self._task(state)}\n</User_Query>
                """
        try:
            answer = self._complete(state, ASB_ALIGNMENT_PROMPT, data)
        except Exception as exc:
            answer = f"{type(exc).__name__}: {exc}"
        enforced = self.settings.mode == "block"
        self._record(
            state,
            "model_output",
            allowed=False,
            enforced=enforced,
            reason=f"DRIFT trajectory deviation {name} was refused",
            source="constraint",
            tool=name,
            detail={"judge": answer},
        )
        if enforced:
            state.setdefault("_drift_asb_denied", []).append([name, arguments])
        elif self.settings.mode == "warn":
            state["_drift_warning"] = f"DRIFT trajectory deviation {name} was refused"

    def _permission(self, state: dict[str, Any], name: str) -> str:
        # DRIFTLLM.py:155-184; classified lazily instead of for every tool up front.
        permissions = state.setdefault("_drift_permissions", {})
        if name not in permissions:
            tool = next(
                item for item in self._tools_docs(state) if item["name"] == name
            )
            data = f"""
                <Function>\n{json.dumps(tool)}\n</Function>
                """
            choice = self._complete(state, PRIVILEGE_PROMPT, data)
            permissions[name] = (
                "Write" if "B" in choice else "Execute" if "C" in choice else "Read"
            )
        return permissions[name]

    def _format_checklist(self, state: dict[str, Any], checklist: str) -> str:
        # node_json_formatting (DRIFTLLM.py:232-286).
        data = f"""
                <User_Query>\n{self._task(state)}\n</User_Query>
                <Parameter_Checklist>\n{checklist}\n</Parameter_Checklist>
                """
        formatted = checklist
        try:
            for _ in range(3):
                formatted = repair_json(
                    self._complete(state, CHECKLIST_FORMAT_PROMPT, data)
                )
                try:
                    json.loads(formatted)
                    break
                except ValueError:
                    continue
        except Exception as exc:
            self._record(
                state, "checklist_format", allowed=True, enforced=False,
                reason=f"{type(exc).__name__}: {exc}", source="error",
            )
            return checklist
        return formatted

    def _task(self, state: dict[str, Any]) -> str:
        return str(state.get("_drift_user_request") or self.cfg.agent.task)

    def _tools(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        supplied = state.get("_drift_tools")
        return (
            copy.deepcopy(supplied)
            if isinstance(supplied, list)
            else tool_definitions_from_config(self.cfg)
        )

    def _tools_docs(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        # achieve_tools (DRIFTLLM.py:562-577): name, description, JSON schema.
        return [
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters")
                or tool.get("inputSchema")
                or tool.get("input_schema")
                or {},
            }
            for tool in self._tools(state)
        ]

    def _complete(self, state: dict[str, Any], system: str, user: str) -> str:
        if self._llm is None:
            base = self.cfg.llm
            settings = self.settings
            self._llm = LLMAdapter(
                replace(
                    base,
                    provider=settings.provider or base.provider,
                    model=settings.model or base.model,
                    temperature=settings.temperature
                    if settings.temperature is not None
                    else base.temperature,
                    base_url=settings.base_url or base.base_url,
                    api_key=settings.api_key or base.api_key,
                    api_key_env=settings.api_key_env or base.api_key_env,
                    request_timeout=settings.request_timeout
                    if settings.request_timeout is not None
                    else base.request_timeout,
                )
            )
        response = self._llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        usage = response.usage or {}
        stats = state.setdefault("trace_stats", {})
        stats["api_calls"] = int(stats.get("api_calls", 0)) + 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            stats[key] = int(stats.get(key, 0)) + int(usage.get(key) or 0)
        return str(response.content or "")

    def _record(
        self,
        state: dict[str, Any],
        phase: str,
        *,
        allowed: bool,
        enforced: bool,
        reason: str,
        source: str,
        tool: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "phase": phase,
            "tool": tool,
            "allowed": allowed,
            "enforced": enforced,
            "reason": reason,
            "source": source,
            "mode": self.settings.mode,
            "profile": self.profile,
            **(detail or {}),
        }
        state["_last_drift_decision"] = event
        events = state.setdefault("drift_events", [])
        events.append(event)
        state.setdefault("harness", {})["drift"] = {
            "enabled": True,
            "mode": self.settings.mode,
            "profile": self.profile,
            "status": "error" if source == "error" else "active",
            "event_count": len(events),
            "last_decision": event,
            "completed_trajectory": list(state.get("_drift_completed") or []),
        }
        self._artifact(state, "drift_events.json", events)
        return event

    def _artifact(self, state: dict[str, Any], filename: str, value: Any) -> None:
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if not run_dir:
            return
        path = Path(str(run_dir)) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
