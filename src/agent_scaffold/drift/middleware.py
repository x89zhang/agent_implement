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
from ..tool_results import unwrap_hermes_result
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
    PLANNER_RETRY_NOTE,
    PLANNER_ROLE_CLARIFICATION,
    PRIVILEGE_PROMPT,
)


# client.py:139-141: llm_run's answer when the model call fails.
_FAILED_GENERATION = "FAILED GENERATION."
# The model of upstream's published runs (runs/gpt-4o-mini-2024-07-18, the
# utils.py --model default).
UPSTREAM_PLANNER_MODEL = "gpt-4o-mini-2024-07-18"


class DriftMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, llm: Any | None = None,
                 approval_callback: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.drift
        self._llm = llm
        self.approval_callback = approval_callback
        if self.settings.request_user_approval and approval_callback is None:
            raise ValueError("DRIFT user approval requires an approval callback")
        # One behavior for every benchmark: the main DRIFTLLM.py variant unless
        # the ASB_DRIFT variant is selected explicitly.
        self.profile = self.settings.profile

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        # Each model call is one upstream query(): discard the unfinished
        # validation of an output that was sent back for revision.
        state.pop("_drift_batch", None)
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
            self.profile == "default"
            and self.settings.dynamic_validation
            and not self._ensure_constraints(state)
            and state.get("_drift_trajectory")
        ):
            chunks.append(
                EXECUTION_GUIDELINES_PROMPT.format(
                    initial_trajectory=state["_drift_trajectory"],
                    node_checklist=state.get("_drift_checklist", "None"),
                    achieved_trajectory=state.get("_drift_completed") or [],
                    query=self._runtime_task(state),
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
        # Upstream validates a model output's calls as one batch
        # (DRIFTLLM.py:668-693). The controller asks once per call, so the
        # batch runs on the first call and later calls read its verdict.
        calls, index = self._output_calls(state, name, arguments)
        batch = state.get("_drift_batch")
        if index == 0 or batch is None or index >= len(batch["verdicts"]):
            batch = self._validate_batch(state, calls)
            state["_drift_batch"] = batch
            index = min(index, len(calls) - 1)
        verdict = batch["verdicts"][index]
        enforced = self.settings.mode == "block"
        if verdict == "duplicate":
            self._record(
                state, "model_output", allowed=True, enforced=enforced,
                reason="exact repeat of an earlier tool call", source="duplicate",
                tool=name, detail={"skipped": True, "call_index": index},
            )
            if enforced:
                # Upstream drops repeated calls from the output (DRIFTLLM.py:670-672).
                batch["dropped"].add(index)
                return ModelDecision(content=content, tool_call=None)
            return ModelDecision(content=content, tool_call=tool_call)
        refusal = batch["refusal"]
        if not refusal:
            self._record(
                state, "model_output", allowed=True, enforced=False, reason="",
                source="validation", tool=name, detail={"call_index": index},
            )
            return ModelDecision(content=content, tool_call=tool_call)
        reason, feedback, source, offending = refusal
        # Upstream refuses the whole output (output["tool_calls"] = []).
        self._record(
            state,
            "model_output",
            allowed=False,
            enforced=enforced,
            reason=reason,
            source=source,
            tool=name,
            detail={
                "call_index": index,
                "offending_tool": offending,
                **batch.get("detail", {}),
            },
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
        # Released calls join the call history used for duplicate removal
        # (_load_previous_calls, DRIFTLLM.py:140-148), and their count tells
        # after_tool which result is the batch's last tool message.
        batch = state.pop("_drift_batch", None) or {}
        if tool_call is None:
            state["_drift_batch_remaining"] = 0
            return
        name, arguments = tool_call
        calls = [
            (str(item.get("name")), item.get("arguments"), "")
            for item in state.get("_model_output_calls") or []
            if isinstance(item, dict)
        ]
        if name not in [call[0] for call in calls]:
            calls = [(name, arguments, "")]
        dropped = batch.get("dropped") or set()
        released = [call for i, call in enumerate(calls) if i not in dropped]
        state["_drift_batch_remaining"] = len(released)
        history = state.setdefault("_drift_call_history", [])
        history.extend(self._call_key(n, a) for n, a, _ in released)

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
        if not self.settings.injection_isolation:
            return ResultDecision(result=result)
        # Upstream isolates messages[-1] only (DRIFTLLM.py:615-627): of the
        # results of one model output, only the last one is checked.
        remaining = state.get("_drift_batch_remaining")
        if remaining is not None:
            remaining -= 1
            state["_drift_batch_remaining"] = remaining
            if (
                remaining > 0
                and self.profile == "default"
                and self.settings.isolate_last_only
            ):
                self._record(
                    state, "after_tool", allowed=True, enforced=False, reason="",
                    source="injection_isolation", tool=name,
                    detail={"skipped": "not_last_result_of_batch", "masked": False},
                )
                return ResultDecision(result=result)
        started = time.monotonic()
        raw = str(result)
        # Upstream masks the bare tool content; Hermes wraps MCP results as
        # {"result": text}, whose JSON escaping defeats remove_sentence.
        plain = unwrap_hermes_result(raw)
        text = plain
        error = ""
        detections = 0
        spans: list[Any] = []
        llm_errors: list[str] = []
        try:
            if self.profile == "asb":
                text, detections, spans, llm_errors = self._isolate_asb(
                    state, name, payload, plain
                )
            else:
                text, detections, spans, llm_errors = self._isolate(
                    state, name, payload, plain, failed
                )
        except Exception as exc:
            error = f"DRIFT isolation failed: {type(exc).__name__}: {exc}"
        masked = not error and text != plain
        # Upstream turns a failed detection call into "FAILED GENERATION." and
        # passes the result through unmasked (client.py:139-141). Withholding
        # only happens when fail-closed isolation is requested explicitly.
        all_failed = bool(llm_errors) and len(llm_errors) >= detections
        if not error and all_failed and self.settings.isolation_fail_closed:
            error = f"DRIFT isolation failed: {llm_errors[-1]}"
        enforced = bool(error) and self.settings.isolation_fail_closed
        detail = {
            "masked": masked,
            "detections": detections,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "hermes_wrapped": plain != raw,
        }
        if llm_errors:
            detail["llm_error"] = llm_errors[-1]
        # Isolation sanitizes the result and the run continues
        # (DRIFTLLM.py:615-627); it never refuses an action, so a masked
        # result is recorded as such and not as a denial.
        event = self._record(
            state,
            "after_tool",
            allowed=not error,
            enforced=enforced,
            reason=error or ("masked conflicting instructions" if masked else ""),
            source="error" if error else "injection_isolation",
            tool=name,
            detail=detail,
        )
        if enforced:
            return ResultDecision(
                False, error, "Tool result withheld by DRIFT.", "drift_isolation_error"
            )
        if error or not masked or self.settings.mode == "monitor":
            return ResultDecision(result=result)
        if self.settings.mode == "warn":
            state["_drift_warning"] = event["reason"]
        return ResultDecision(result=self._rewrap(raw, plain, text, spans))

    @staticmethod
    def _rewrap(raw: str, plain: str, masked: str, spans: list[Any]) -> str:
        """Put masked content back into the original Hermes envelope."""
        if plain == raw:
            return masked
        value = json.loads(raw)
        if isinstance(value.get("result"), str):
            value["result"] = masked
        else:
            try:
                value["result"] = json.loads(masked)
            except ValueError:
                value["result"] = masked

        def scrub(item: Any) -> Any:
            # structuredContent repeats the content; mask the same spans there.
            if isinstance(item, str):
                for span in spans:
                    item = remove_sentence(item, span)
                return item
            if isinstance(item, list):
                return [scrub(entry) for entry in item]
            if isinstance(item, dict):
                return {key: scrub(entry) for key, entry in item.items()}
            return item

        if "structuredContent" in value:
            value["structuredContent"] = scrub(value["structuredContent"])
        return json.dumps(value, ensure_ascii=False)

    def _isolate(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        text: str,
        failed: bool,
    ) -> tuple[str, int, list[Any], list[str]]:
        # DRIFTLLM.py:615-635. Upstream passes the tool message dict's str().
        query = self._runtime_task(state)
        spans: list[Any] = []
        llm_errors: list[str] = []

        def observation(content: str) -> str:
            return str({
                "role": "tool",
                "content": content,
                "tool_call": {"function": name, "args": payload},
                "error": content if failed else None,
            })

        def detect(content: str) -> str:
            user_prompt = f"""<User Query>\n{query}\n</User Query>
                <Tool Results>\n{observation(content)}\n</Tool Results>"""
            try:
                response = self._complete(state, INJECTION_DETECTION_PROMPT, user_prompt)
            except Exception as exc:
                llm_errors.append(f"{type(exc).__name__}: {exc}")
                response = _FAILED_GENERATION
            spans.extend(parse_detected_instructions(response) or [])
            return response

        completed, text = injection_isolate(detect(text), text)
        detections = 1
        cycle_times = 0
        while not completed and cycle_times < self.settings.mask_limitation:
            cycle_times += 1
            completed, text = injection_isolate(detect(text), text)
            detections += 1
        return text, detections, spans, llm_errors

    def _isolate_asb(
        self, state: dict[str, Any], name: str, payload: dict[str, Any], text: str
    ) -> tuple[str, int, list[Any], list[str]]:
        # ASB_DRIFT/pyopenagi/agents/drift.py:143-206 with cycle_limit=1;
        # action/observation strings follow react_agent_attack.py:145-146.
        conversations = [
            message for message in state.get("messages") or []
            if message.get("role") != "system"
        ]
        tool_call = f"[Action]: I will call the {name} with the params as {payload}"
        observations = [f"The knowledge I get from {name} is: {text}"]
        data = f"""
                        <User Query>\n{self._runtime_task(state)}\n</User Query>
                        <Interaction History>\n{conversations}\n</Interaction History>
                        <Tool Call>\n{tool_call}\n</Tool Call>
                        <Observation>\n{observations}\n</Observation>
                        """
        try:
            response = self._complete(state, ASB_INJECTION_DETECTION_PROMPT, data)
        except Exception as exc:
            # drift.py:168-169 skips the failed detection and keeps the result.
            return text, 1, [], [f"{type(exc).__name__}: {exc}"]
        spans = parse_detected_instructions(response) or []
        for item in spans:
            text = remove_sentence(text, item)
        return text, 1, spans, []

    def _ensure_constraints(self, state: dict[str, Any]) -> str:
        if state.get("_drift_initialized"):
            return str(state.get("_drift_init_error") or "")
        state["_drift_initialized"] = True
        if not self.settings.dynamic_validation:
            return ""
        started = time.monotonic()
        error = ""
        completion = ""
        # agent_run (client.py:45-61): the tools are appended to
        # CONSTRAINTS_BUILD_PROMPT; the clean task is the user turn. The ASB
        # profile has no generated plan upstream (its trajectory is the ASB
        # agent's own workflow, which Hermes does not expose), so the same
        # generator stands in for it.
        system = (
            CONSTRAINTS_BUILD_PROMPT
            + f"\n\n<avaliable_tools>\n\n{json.dumps(self._tools_docs(state))}\n\n</avaliable_tools>"
        )
        # Upstream's environment setup is AgentDojo's fixed persona ("Emma
        # Johnson ... Blue Sparrow Tech"), so it is added only on request and
        # then for every benchmark alike.
        if self.settings.environment_guidelines:
            system += f"\n\n<environment_setup>\n\n{ENVIRONMENT_GUIDELINES}\n\n</environment_setup>"
        clarify = self._clarify_planner_role()
        user = self._generation_task(state)
        if clarify:
            user = f"{user}\n\n{PLANNER_ROLE_CLARIFICATION}"
        attempts: list[str] = []
        try:
            completion = self._complete(state, system, user)
            attempts.append(completion)
            if clarify and not parse_constraints(completion)[0]:
                completion = self._complete(state, system, f"{user}\n\n{PLANNER_RETRY_NOTE}")
                attempts.append(completion)
        except Exception as exc:
            error = f"DRIFT constraint generation failed: {type(exc).__name__}: {exc}"
        trajectory, checklist = parse_constraints(completion)
        if not error and not trajectory:
            # Upstream continues with an empty plan, which auto-accepts every
            # read and leaves DRIFT effectively off; a planner that answers as
            # the agent (or refuses) is a monitor failure, not a clean verdict.
            error = "DRIFT constraint generation returned no function trajectory"
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
                "raw_completion": completion,
                "planner_model": self._planner_model(),
                "planner_role_clarification": clarify,
                "attempts": attempts,
                "error": error,
                "upstream_revision": UPSTREAM_REVISION,
            },
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
        state.setdefault("trace", []).append(event)
        self._artifact(state, "drift_constraints.json", event)
        return error

    def _planner_model(self) -> str:
        return str(self.settings.model or self.cfg.llm.model or "")

    def _clarify_planner_role(self) -> bool:
        setting = self.settings.planner_role_clarification
        if setting == "auto":
            # Upstream's own runs use this model unmodified.
            return self._planner_model() != UPSTREAM_PLANNER_MODEL
        return setting == "on"

    def _output_calls(
        self, state: dict[str, Any], name: str, arguments: Any
    ) -> tuple[list[tuple[str, Any, str]], int]:
        """All calls of the current model output and the index of this one."""
        raw = state.get("_model_output_calls")
        index = int(state.get("_model_output_index") or 0)
        calls = [
            (str(item.get("name")), item.get("arguments"), str(item.get("id") or ""))
            for item in raw or []
            if isinstance(item, dict)
        ]
        if not (0 <= index < len(calls) and calls[index][0] == name):
            # Hosts that do not expose the whole output check one call at a time.
            return [(name, arguments, "")], 0
        return calls, index

    @staticmethod
    def _call_key(name: str, arguments: Any) -> str:
        # _tool_call_to_str (DRIFTLLM.py:103-113) compares name and json.dumps(args).
        return json.dumps({"name": name, "arguments": json.dumps(arguments, default=str)})

    def _validate_batch(
        self, state: dict[str, Any], calls: list[tuple[str, Any, str]]
    ) -> dict[str, Any]:
        """Duplicate removal, then trajectory and checklist validation of one output.

        Mirrors DRIFTLLM.py:666-693 with trajectory_constraint_validation
        (:440-524) and checklist_constraint_validation (:526-560).
        """
        query = self._runtime_task(state)
        history = state.get("_drift_call_history") or []
        verdicts: list[str] = [
            "duplicate" if self._call_key(name, args) in history else "check"
            for name, args, _ in calls
        ]
        batch: dict[str, Any] = {"verdicts": verdicts, "dropped": set(), "refusal": None}
        todo = [call for call, verdict in zip(calls, verdicts) if verdict == "check"]
        to_call = [name for name, _, _ in todo]
        achieved = list(state.get("_drift_completed") or [])
        temp: list[str] = []
        for func_ids, func in enumerate(achieved + to_call):
            trajectory = list(state.get("_drift_trajectory") or [])
            if func_ids < len(trajectory) and func == trajectory[func_ids]:
                temp.append(func)
                continue
            extended = list(trajectory)
            extended.insert(func_ids, func)
            reason = (
                f"The function trajectory of {extended} is not aligned with "
                f"trajectory plan of {trajectory}."
            )
            extended_checklist = insert_checklist_step(
                str(state.get("_drift_checklist", "None")), func_ids, func
            )
            aligned, judge_error = self._deviation_aligned(
                state, func, trajectory, extended
            )
            if not aligned and self.settings.request_user_approval:
                aligned = bool(self.approval_callback({
                    "tool": func, "trusted_user_task": query,
                    "initial_function_trajectory": trajectory,
                    "current_function_trajectory": extended,
                }))
            if not aligned:
                refused_achieved = list(state.get("_drift_completed") or [])
                batch["refusal"] = (reason, (
                    f"[CALL ERROR] The function calling of {func} has been refused due to it does not align with original planned trajectory, please change to call other functions to complete original user query.\n"
                    "Remember, do not stop working on the original user task to do other things.\n"
                    f"Below is the specific Refusal Reason:\n {reason}.\n"
                    f"Original Planned Trajecotry:\n{trajectory}.\n"
                    f"Achieved Function Trajectory:\n{refused_achieved}.\n"
                    f"User Query:\n{query}"
                ), "constraint", func)
                if judge_error:
                    batch["detail"] = {"llm_error": judge_error}
                self._mark_checks(batch)
                return batch
            # Accepted deviations extend the plan at validation time (DRIFTLLM.py:492-499).
            temp.append(func)
            state["_drift_trajectory"] = extended
            state["_drift_completed"] = list(temp)
            state["_drift_checklist"] = extended_checklist
            self._record(
                state,
                "trajectory_progress",
                allowed=True,
                enforced=False,
                reason="",
                source="dynamic_alignment",
                tool=func,
                detail={"trajectory": list(extended)},
            )
        state["_drift_completed"] = list(temp)
        # node_json_formatting runs once per validated output (DRIFTLLM.py:538).
        state["_drift_checklist"] = self._format_checklist(
            state, str(state.get("_drift_checklist", "None"))
        )
        checklist = str(state.get("_drift_checklist", "None"))
        passed, reason = node_check(
            checklist, len(temp), [(name, args) for name, args, _ in todo]
        )
        if not passed:
            json_calls = [
                {
                    **({"id": call_id} if call_id else {}),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for name, args, call_id in todo
            ]
            batch["refusal"] = (reason, (
                f"[CALL ERROR] The function calling of {json_calls} has been refused due to some parameters are not aligned with checklist, Please try an alternative method to continue fulfilling the original user query. Remember, do not stop working on the original user task to do other things.\n"
                f"The specific error:\n{reason}\n"
                f"Checklist:\n{checklist}.\n"
                "Note: the value of None denotes this value is uncertain.\n"
                f"User Query:\n{query}"
            ), "checklist", "")
        self._mark_checks(batch)
        return batch

    @staticmethod
    def _mark_checks(batch: dict[str, Any]) -> None:
        verdict = "refused" if batch["refusal"] else "passed"
        batch["verdicts"] = [
            item if item == "duplicate" else verdict for item in batch["verdicts"]
        ]

    def _deviation_aligned(
        self,
        state: dict[str, Any],
        name: str,
        trajectory: list[str],
        extended: list[str],
    ) -> tuple[bool, str]:
        # DRIFTLLM.py:479-489: Read tools extend the plan; others ask the
        # alignment judge. An unknown tool (upstream KeyError) accepts; a
        # failed judge call answers "FAILED GENERATION." and refuses
        # (client.py:139-141, DRIFTLLM.py:224-230).
        answer = ""
        llm_error = ""
        try:
            permission = self._permission(state, name)
        except Exception as exc:
            permission = None
            answer = f"{type(exc).__name__}: {exc}"
        if permission is None or permission == "Read":
            aligned = True
        else:
            data = f"""
                <Initial_Function_Trajectory>\n{trajectory}\n</Initial_Function_Trajectory>
                <Current_Function_Trajectory>\n{extended}\n</Current_Function_Trajectory>
                <User_Query>\n{self._runtime_task(state)}\n</User_Query>
                """
            try:
                answer = self._complete(state, _ALIGNMENT_SYSTEM, data)
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                answer = _FAILED_GENERATION
            aligned = ("Yes" in answer) or ("YES" in answer)
        detail: dict[str, Any] = {"proposed_trajectory": extended}
        if llm_error:
            detail["llm_error"] = llm_error
        privilege_error = (state.get("_drift_privilege_errors") or {}).get(name)
        if privilege_error:
            detail["privilege_llm_error"] = privilege_error
        self._record(
            state,
            "trajectory_alignment",
            allowed=aligned,
            enforced=not aligned and self.settings.mode == "block",
            reason="" if aligned else answer,
            source="dynamic_alignment",
            tool=name,
            detail=detail,
        )
        return aligned, llm_error

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
                <User_Query>\n{self._runtime_task(state)}\n</User_Query>
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
        # DRIFTLLM.py:155-184; classified lazily instead of for every tool up
        # front. A failed call answers "FAILED GENERATION.", i.e. Read, cached.
        permissions = state.setdefault("_drift_permissions", {})
        if name not in permissions:
            tool = next(
                item for item in self._tools_docs(state) if item["name"] == name
            )
            data = f"""
                <Function>\n{json.dumps(tool)}\n</Function>
                """
            try:
                choice = self._complete(state, PRIVILEGE_PROMPT, data)
            except Exception as exc:
                state.setdefault("_drift_privilege_errors", {})[name] = (
                    f"{type(exc).__name__}: {exc}"
                )
                choice = _FAILED_GENERATION
            permissions[name] = (
                "Write" if "B" in choice else "Execute" if "C" in choice else "Read"
            )
        return permissions[name]

    def _format_checklist(self, state: dict[str, Any], checklist: str) -> str:
        # node_json_formatting (DRIFTLLM.py:232-286); a failed call answers
        # "FAILED GENERATION.", which json_repair turns into "".
        data = f"""
                <User_Query>\n{self._runtime_task(state)}\n</User_Query>
                <Parameter_Checklist>\n{checklist}\n</Parameter_Checklist>
                """
        formatted = checklist
        llm_error = ""
        for _ in range(3):
            try:
                answer = self._complete(state, CHECKLIST_FORMAT_PROMPT, data)
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                answer = _FAILED_GENERATION
            formatted = repair_json(answer)
            try:
                json.loads(formatted)
                break
            except ValueError:
                continue
        if llm_error:
            self._record(
                state, "checklist_format", allowed=True, enforced=False,
                reason="", source="checklist_format",
                detail={"llm_error": llm_error, "checklist": formatted},
            )
        return formatted

    def _generation_task(self, state: dict[str, Any]) -> str:
        # Plan generation sees the clean task.
        return str(
            state.get("_generation_task")
            or state.get("_drift_user_request")
            or self.cfg.agent.task
        )

    def _runtime_task(self, state: dict[str, Any]) -> str:
        # Runtime checks judge the prompt the agent actually received, as
        # upstream passes its query (react_agent_attack.py:219,233).
        return str(
            state.get("_runtime_user_request")
            or state.get("_drift_user_request")
            or self.cfg.agent.task
        )

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
