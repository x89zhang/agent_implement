"""Execute the upstream Hermes agent in its own interpreter and profile.

This launcher supplies tools/configuration and records output; Hermes owns the loop.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from importlib.metadata import distributions
from pathlib import Path

from defense_plugin import DefensePlugin, GuardStopped, ReplayRecorderPlugin


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    for key in (
        "BENCHMARK_MODEL_API_KEY",
        "BENCHMARK_BRIDGE_TOKEN",
        "BENCHMARK_GUARD_TOKEN",
    ):
        secret = os.environ.get(key, "")
        if secret:
            text = text.replace(secret, "***REDACTED***")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def install_opaque_home_alias():
    """Hide benchmark and condition names embedded in the artifact path."""
    original = os.environ.get("HERMES_HOME", "").strip()
    if not original:
        return None
    alias = Path(tempfile.gettempdir()) / f"hermes-home-{uuid.uuid4().hex}"
    alias.symlink_to(Path(original), target_is_directory=True)
    os.environ["HERMES_HOME"] = str(alias)
    return alias


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    result_path = Path(sys.argv[2])
    events_path = result_path.parent / "events.jsonl"
    secret = os.environ.get("BENCHMARK_MODEL_API_KEY", "")
    sys.path.insert(0, request["repo_path"])
    agent = None
    result = None
    plugin = None
    home_alias = None

    def event(kind, payload):
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"type": kind, "timestamp": time.time(), "payload": payload},
                    default=str,
                )
                + "\n"
            )

    try:
        if request.get("schema_version") != 1:
            raise ValueError("Unsupported Hermes worker request schema")
        home_alias = install_opaque_home_alias()
        write_json(
            result_path.parent / "dependencies.json",
            {d.metadata["Name"]: d.version for d in distributions()},
        )
        from run_agent import AIAgent
        from tools.mcp_tool_discovery import discover_mcp_tools

        registered = discover_mcp_tools(allowed_mcp_names=["benchmark"])
        if len(registered) != request["tool_count"]:
            raise RuntimeError(
                f"MCP discovery mismatch: expected {request['tool_count']}, got {len(registered)}"
            )
        from tools.mcp_tool_schema import mcp_prefixed_tool_name

        mapping = {
            mcp_prefixed_tool_name("benchmark", name): name
            for name in request["tool_names"]
        }
        if set(mapping) != set(registered):
            raise RuntimeError("MCP canonical tool-name mapping mismatch")
        write_json(result_path.parent / "tool_name_map.json", mapping)
        toolsets = ["mcp-benchmark"]
        if request["memory_enabled"]:
            toolsets.append("memory")
        if request.get("skills"):
            toolsets.append("skills")
        agent = AIAgent(
            model=request["model"],
            provider=request["provider"],
            api_mode=None
            if request.get("api_mode", "auto") == "auto"
            else request["api_mode"],
            base_url=request["base_url"] or None,
            api_key=secret or None,
            max_iterations=request["max_iterations"],
            run_budget_seconds=request["timeout_seconds"],
            enabled_toolsets=toolsets,
            quiet_mode=True,
            session_id=str(uuid.uuid4()),
            platform="cli",
            skip_context_files=True,
            load_soul_identity=False,
            skip_memory=not request["memory_enabled"],
            skip_background_review=True,
            event_callback=event,
            ephemeral_system_prompt=(
                "Use skill_view to read the following task skills before acting: "
                + ", ".join(s["name"] for s in request["skills"])
            )
            if request.get("skills")
            else None,
            # Hermes resolves ``api_mode: auto`` during construction. Responses-only GPT
            # models reject Chat Completions sampling fields; apply temperature after routing.
            request_overrides={},
        )
        if agent.api_mode == "chat_completions":
            agent.request_overrides["temperature"] = request["temperature"]
        write_json(
            result_path.parent / "transport.json",
            {
                "api_mode": agent.api_mode,
                "provider": agent.provider,
                "model": agent.model,
            },
        )
        actual = {tool["function"]["name"] for tool in agent.tools}
        expected = set(registered) | (
            {"memory"} if request["memory_enabled"] else set()
        )
        if request.get("skills"):
            expected |= {"skills_list", "skill_view", "skill_manage"}
        if actual != expected:
            raise RuntimeError(
                f"Unexpected model-visible tools: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        write_json(result_path.parent / "tools.json", agent.tools)
        event("tool_inventory", {"names": sorted(actual)})
        if request.get("defense_mode") == "replay":
            plugin = ReplayRecorderPlugin(
                agent,
                request,
                mapping,
                event,
                result_path.parent / "guard_lifecycle.jsonl",
            )
        else:
            plugin = DefensePlugin(agent, request, mapping, event)
        plugin.install()
        raw = agent.run_conversation(request["prompt"], conversation_history=None)
        write_json(
            result_path.parent / "system_prompt.json",
            {"system_prompt": getattr(agent, "_cached_system_prompt", None)},
        )
        completed = bool(raw.get("completed")) and not raw.get("failed")
        result = {
            "schema_version": 1,
            "status": "completed" if completed else "failed",
            "final_output": raw.get("final_response") or "",
            "messages": raw.get("messages") or [],
            "api_calls": raw.get("api_calls"),
            "raw_result": raw,
            "termination_reason": raw.get("error") or raw.get("termination_reason"),
        }
    except GuardStopped as exc:
        event("defense_stop", {"reason": exc.reason, "failed": exc.failed})
        result = {
            "schema_version": 1,
            "status": "failed" if exc.failed else "completed",
            "final_output": exc.output,
            "messages": plugin.messages if plugin else [],
            "termination_reason": exc.reason,
        }
    except Exception as exc:  # noqa: BLE001 - serialize failures at the process/tool boundary
        result = {
            "schema_version": 1,
            "status": "failed",
            "final_output": "",
            "messages": [],
            "termination_reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            if agent is not None:
                agent.close()
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers

            shutdown_mcp_servers()
        except Exception as exc:  # noqa: BLE001 - serialize failures at the process/tool boundary
            if result is not None:
                result["status"] = "failed"
                result["termination_reason"] = (
                    f"Hermes cleanup failed: {type(exc).__name__}: {exc}"
                )
        if home_alias is not None:
            try:
                home_alias.unlink(missing_ok=True)
            except OSError:
                pass
        # Shutdown is inside the supervisor's deadline and precedes the memory snapshot.
        write_json(result_path, result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
