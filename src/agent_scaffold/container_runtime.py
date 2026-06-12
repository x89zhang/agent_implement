from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def should_run_in_container(cfg: Any) -> bool:
    if os.environ.get("AGENT_CONTAINERIZED") == "1":
        return False
    container = getattr(cfg, "container", None)
    return bool(getattr(container, "enabled", False))


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def _workspace_container_path(path: Path, workspace_root: Path, container_workdir: str) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside workspace for container execution: {resolved}") from exc
    return str(Path(container_workdir) / rel)


def _run_checked(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def _ensure_image(cfg: Any, workspace_root: Path) -> None:
    image = str(cfg.container.image)
    inspect = _run_checked(["docker", "image", "inspect", image], workspace_root)
    if inspect.returncode == 0:
        return
    if not bool(cfg.container.auto_build):
        raise RuntimeError(
            f"Container image {image!r} was not found and container.auto_build is disabled. "
            "Build it first or set container.enabled: false."
        )
    dockerfile = Path(str(cfg.container.dockerfile))
    dockerfile_path = dockerfile if dockerfile.is_absolute() else workspace_root / dockerfile
    build_cmd = ["docker", "build", "-t", image, "-f", str(dockerfile_path)]
    for key, value in (getattr(cfg.container, "build_args", {}) or {}).items():
        build_cmd.extend(["--build-arg", f"{key}={value}"])
    build_cmd.append(str(workspace_root))
    build = _run_checked(build_cmd, workspace_root)
    if build.returncode != 0:
        raise RuntimeError(
            "Failed to build agent container image.\n"
            f"Command: {' '.join(build_cmd)}\n"
            f"STDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )


def run_once_in_container(
    cfg: Any,
    cfg_path: str,
    user_input: str | None,
    context_messages: list[dict[str, str]] | None,
    resume_messages: list[dict[str, str]] | None,
    workspace_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    _ensure_image(cfg, workspace_root)

    container_workdir = str(cfg.container.workdir).rstrip("/") or "/workspace"
    config_in_container = _workspace_container_path(Path(cfg_path), workspace_root, container_workdir)
    run_dir_in_container = _workspace_container_path(run_dir, workspace_root, container_workdir)

    payload_path = run_dir / "_container_payload.json"
    result_path = run_dir / "_container_result.json"
    stdout_path = run_dir / "container_stdout.log"
    stderr_path = run_dir / "container_stderr.log"
    payload_path.write_text(
        json.dumps(
            {
                "user_input": user_input,
                "context_messages": context_messages,
                "resume_messages": resume_messages,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    cmd = ["docker", "run"]
    if bool(cfg.container.remove):
        cmd.append("--rm")
    network = str(cfg.container.network or "").strip()
    if network:
        cmd.extend(["--network", network])
    cmd.extend(["-v", f"{workspace_root}:{container_workdir}", "-w", container_workdir])
    cmd.extend(["-e", "PYTHONPATH=src", "-e", "AGENT_CONTAINERIZED=1"])
    cmd.extend(["-e", f"AGENT_JOB_DIR={run_dir_in_container}"])
    cmd.extend(["-e", f"AGENT_RESULT_PATH={run_dir_in_container}/_container_result.json"])
    for name in getattr(cfg.container, "env", []) or []:
        value = os.environ.get(str(name))
        if value is not None:
            cmd.extend(["-e", f"{name}={value}"])
    cmd.append(str(cfg.container.image))
    cmd.extend(
        [
            sys.executable.split("/")[-1] if sys.executable else "python",
            "src/agent_scaffold/main.py",
            "--config",
            config_in_container,
            "--run-payload",
            f"{run_dir_in_container}/_container_payload.json",
        ]
    )

    completed = _run_checked(cmd, workspace_root)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        error = {
            "error": "container_run_failed",
            "returncode": completed.returncode,
            "command": cmd,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        (run_dir / "container_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Container run failed with exit code {completed.returncode}. "
            f"See {stderr_path} and {stdout_path}."
        )

    if not result_path.exists():
        raise RuntimeError(f"Container completed but did not write result file: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))
