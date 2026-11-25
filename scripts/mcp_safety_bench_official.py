"""Upstream-side helper for the MCP-SafetyBench adapter.

Runs with the MCP-SafetyBench interpreter (the upstream requirements) and the
pinned source tree on PYTHONPATH; the project adapter calls it as a subprocess.

  additions TASK_JSON   Serve the task's added tool (mcp_server_additions) over
                        stdio. Upstream inserts `@mcp.tool()` + the task's
                        implementation_code into a FastMCP server
                        (benchmark/runner.py _inject_malicious_tools); this
                        registers the same code the same way, so the tool's name,
                        description (its docstring) and input schema match.
  evaluate              Read {"task_path", "result", "calls"} as JSON on stdin and
                        print the upstream Task evaluations as JSON on stdout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from types import SimpleNamespace

# Upstream marks these comparison ops as trace checks (evaluator/evaluator.py).
TRACE_OPS = {
    "check_malicious_tool_attack",
    "check_dependency",
    "check_parameter_modification_attack",
    "check_command_injection_attack",
    "check_rug_pull_attack",
    "check_wrong_function_intent_injection",
    "check_wrong_arguments_intent_injection",
    "check_data_tampering_attack",
    "check_identity_spoofing_attack",
    "check_replay_injection_attack",
}


def additions(task_json: str) -> None:
    from mcp.server.fastmcp import FastMCP

    with open(task_json, encoding="utf-8") as stream:
        task = json.load(stream)
    added = task.get("mcp_server_additions") or {}
    tools = [added] if "tool_name" in added else list(added.get("tools", []))
    server = FastMCP("mcp-safety-bench-additions")
    for tool in tools:
        namespace: dict = {}
        # dont_inherit: this module's `from __future__ import annotations` would
        # turn the tool's annotations into strings, which FastMCP 1.9 rejects.
        code = compile(tool["implementation_code"], tool["tool_name"], "exec", dont_inherit=True)
        exec(code, namespace)  # noqa: S102 - benchmark-provided tool code
        server.tool()(namespace[tool["tool_name"]])
    server.run(transport="stdio")


def _trace_records(calls: list[dict]) -> list:
    """Upstream TraceRecord shape as read by the evaluator functions."""
    records = [
        SimpleNamespace(
            timestamp=float(call["timestamp"]),
            data={
                "type": "tool",
                "class": "MCPClient",
                "server": call.get("server", ""),
                "tool_name": call["tool_name"],
                "arguments": call.get("arguments") or {},
                "response": call.get("response", ""),
                "error": call.get("error", ""),
            },
        )
        for call in calls
    ]
    return [SimpleNamespace(records=records)] if records else []


async def _evaluate(request: dict) -> dict:
    from mcpuniverse.benchmark.task import Task
    from mcpuniverse.evaluator.functions import COMPARISON_FUNCTIONS

    task = Task(request["task_path"])
    x = {"result": request["result"], "trace_records": _trace_records(request["calls"])}
    results = await task.evaluate(x)
    evaluations = [
        {
            "func": r.config.func,
            "op": r.config.op,
            "op_args": r.config.op_args,
            "desc": r.config.desc,
            "passed": bool(r.passed),
            "reason": r.reason,
            "error": r.error[:2000],
        }
        for r in results
    ]
    # Upstream (benchmark/runner.py) takes the last evaluator as the attack check
    # and the others as task checks. Also run that check on the trace directly,
    # independent of the evaluator's `json` step on the final answer (which falls
    # back to the raw input when the answer is not JSON); the two should agree.
    attack = task._config.evaluators[-1] if task._config.evaluators else None
    trace_check = None
    if attack is not None and attack.op in TRACE_OPS:
        passed, reason = await COMPARISON_FUNCTIONS[attack.op](
            x=x, op_args=attack.op_args, context=task._context
        )
        trace_check = {"op": attack.op, "passed": bool(passed), "reason": reason}
    return {"evaluations": evaluations, "attack_trace_check": trace_check}


def evaluate() -> None:
    request = json.load(sys.stdin)
    # Upstream evaluator functions print diagnostics; keep stdout for the result.
    with contextlib.redirect_stdout(sys.stderr):
        result = asyncio.run(_evaluate(request))
    json.dump(result, sys.stdout, default=str)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "additions":
        additions(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "evaluate":
        evaluate()
    else:
        raise SystemExit("usage: mcp_safety_bench_official.py additions TASK_JSON | evaluate")


if __name__ == "__main__":
    main()
