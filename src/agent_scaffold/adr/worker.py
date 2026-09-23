"""Run the upstream ADR detector in its own Python environment.

Protocol: one JSON request on stdin, one JSON response on stdout. Upstream
logging and print calls are redirected to stderr so the protocol stays clean.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path


def main() -> int:
    try:
        request = json.load(sys.stdin)
        detection_root = Path(request["detection_root"]).resolve()
        if not (detection_root / "guardrail" / "adr_agent" / "adr_baseline.py").is_file():
            raise ValueError("ADR detection_root must point to the upstream Detection directory")
        sys.path.insert(0, str(detection_root))
        with contextlib.redirect_stdout(sys.stderr):
            from guardrail.adr_agent.adr_baseline import ADRBaseline

            detector = ADRBaseline(
                config_data=request["config"],
                benchmark_type=request["benchmark_type"],
            )
            if not detector.is_available():
                raise RuntimeError("ADR reasoning agent is not ready")
            # Upstream's generated MCP config launches `uv run`, which would
            # sync the entire benchmark project on every fresh container.
            # These three context providers only need this isolated runtime.
            reasoning = detector.reasoning_agent
            mcp_servers = {
                server["name"]: {
                    "command": sys.executable,
                    "args": [server["path"]],
                }
                for server in reasoning.available_servers
            }
            (reasoning.workspace / ".mcp.json").write_text(
                json.dumps({"mcpServers": mcp_servers}), encoding="utf-8"
            )
            result = detector.analyze_task(request["task"])
        print(json.dumps({"ok": True, "result": result.to_dict()}, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
