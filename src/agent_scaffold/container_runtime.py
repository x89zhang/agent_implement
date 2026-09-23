from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .agentdojo_adapter import redact_config_snapshot
from .agentsight import AgentSightObserver
from .config import load_config_mapping


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


def _agentspec_required(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "agentspec", None), "enabled", False))


def _image_has_agentspec(image: str, workspace_root: Path) -> bool:
    probe = _run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            image,
            "-c",
            (
                "import importlib.util as u; "
                "assert u.find_spec('rule') is not None; "
                "assert u.find_spec('interpreter') is not None"
            ),
        ],
        workspace_root,
    )
    return probe.returncode == 0


def _image_has_adr(image: str, workspace_root: Path) -> bool:
    probe = _run_checked(
        [
            "docker", "run", "--rm", "--network", "none",
            "--entrypoint", "/opt/adr-venv/bin/python", image, "-c",
            (
                "from pathlib import Path; import shutil, sys; "
                "assert shutil.which('claude'); "
                "root = Path('/opt/adr/Detection'); "
                "assert (root / 'guardrail/adr_agent/adr_baseline.py').is_file(); "
                "sys.path.insert(0, str(root)); "
                "from guardrail.adr_agent.adr_baseline import ADRBaseline"
            ),
        ],
        workspace_root,
    )
    return probe.returncode == 0


def _image_has_airguard(image: str, workspace_root: Path) -> bool:
    probe = _run_checked(
        [
            "docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
            image, "-c",
            "import sys; sys.path.insert(0, '/opt/airguard/src'); import airguard.guard",
        ],
        workspace_root,
    )
    return probe.returncode == 0


def _image_has_safeagent(image: str, workspace_root: Path) -> bool:
    probe = _run_checked(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
         image, "-c", "import fastmcp"],
        workspace_root,
    )
    return probe.returncode == 0


def _clawsentry_managed(cfg: Any) -> bool:
    settings = getattr(cfg, "clawsentry", None)
    return bool(getattr(settings, "enabled", False) and getattr(settings, "auto_start", True))


def _image_has_clawsentry(image: str, workspace_root: Path) -> bool:
    probe = _run_checked(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/opt/clawsentry-venv/bin/python",
         image, "-c", "from importlib.metadata import version; assert version('clawsentry') == '0.8.7'"],
        workspace_root,
    )
    return probe.returncode == 0


def _effective_build_args(cfg: Any) -> dict[str, str]:
    build_args = {
        str(key): str(value)
        for key, value in (getattr(cfg.container, "build_args", {}) or {}).items()
    }
    if _agentspec_required(cfg):
        build_args["INSTALL_AGENTSPEC"] = "true"
    if getattr(getattr(cfg, "adr", None), "enabled", False):
        build_args["INSTALL_ADR"] = "true"
    if getattr(getattr(cfg, "airguard", None), "enabled", False):
        build_args["INSTALL_AIRGUARD"] = "true"
    if getattr(getattr(cfg, "safeagent", None), "enabled", False):
        build_args["INSTALL_SAFEAGENT"] = "true"
    if _clawsentry_managed(cfg):
        build_args["INSTALL_CLAWSENTRY"] = "true"
    return build_args


def _ensure_image(cfg: Any, workspace_root: Path) -> None:
    image = str(cfg.container.image)
    inspect = _run_checked(["docker", "image", "inspect", image], workspace_root)
    image_exists = inspect.returncode == 0
    agentspec_missing = (
        image_exists
        and _agentspec_required(cfg)
        and not _image_has_agentspec(image, workspace_root)
    )
    adr_missing = (
        image_exists
        and getattr(getattr(cfg, "adr", None), "enabled", False)
        and not _image_has_adr(image, workspace_root)
    )
    airguard_missing = (
        image_exists
        and getattr(getattr(cfg, "airguard", None), "enabled", False)
        and not _image_has_airguard(image, workspace_root)
    )
    safeagent_missing = (
        image_exists
        and getattr(getattr(cfg, "safeagent", None), "enabled", False)
        and not _image_has_safeagent(image, workspace_root)
    )
    clawsentry_missing = (
        image_exists and _clawsentry_managed(cfg)
        and not _image_has_clawsentry(image, workspace_root)
    )
    if image_exists and not any((agentspec_missing, adr_missing, airguard_missing, safeagent_missing, clawsentry_missing)):
        return
    if not bool(cfg.container.auto_build):
        if agentspec_missing:
            raise RuntimeError(
                f"Container image {image!r} does not include the enabled AgentSpec "
                "runtime and container.auto_build is disabled. Build it with "
                "--build-arg INSTALL_AGENTSPEC=true or use a compatible image."
            )
        if adr_missing:
            raise RuntimeError(
                f"Container image {image!r} does not include the enabled ADR "
                "runtime and container.auto_build is disabled. Build it with "
                "--build-arg INSTALL_ADR=true or use a compatible image."
            )
        if airguard_missing:
            raise RuntimeError(
                f"Container image {image!r} lacks AIRGuard source. "
                "Build it with --build-arg INSTALL_AIRGUARD=true."
            )
        if safeagent_missing:
            raise RuntimeError(
                f"Container image {image!r} lacks the SafeAgent MCP client. "
                "Build it with --build-arg INSTALL_SAFEAGENT=true."
            )
        if clawsentry_missing:
            raise RuntimeError(
                f"Container image {image!r} lacks ClawSentry 0.8.7. "
                "Build it with --build-arg INSTALL_CLAWSENTRY=true."
            )
        raise RuntimeError(
            f"Container image {image!r} was not found and container.auto_build is disabled. "
            "Build it first or set container.enabled: false."
        )

    dockerfile = Path(str(cfg.container.dockerfile))
    dockerfile_path = (
        dockerfile if dockerfile.is_absolute() else workspace_root / dockerfile
    )
    build_cmd = ["docker", "build", "-t", image, "-f", str(dockerfile_path)]
    for key, value in _effective_build_args(cfg).items():
        build_cmd.extend(["--build-arg", f"{key}={value}"])
    build_cmd.append(str(workspace_root))
    build = _run_checked(build_cmd, workspace_root)
    if build.returncode != 0:
        raise RuntimeError(
            "Failed to build agent container image.\n"
            f"Command: {' '.join(build_cmd)}\n"
            f"STDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )
    if _agentspec_required(cfg) and not _image_has_agentspec(image, workspace_root):
        raise RuntimeError(
            f"Container image {image!r} was built without an importable AgentSpec "
            "runtime. Ensure its Dockerfile honors INSTALL_AGENTSPEC=true."
        )
    if getattr(getattr(cfg, "adr", None), "enabled", False) and not _image_has_adr(image, workspace_root):
        raise RuntimeError(
            f"Container image {image!r} was built without a working ADR "
            "runtime. Ensure its Dockerfile honors INSTALL_ADR=true."
        )
    if getattr(getattr(cfg, "airguard", None), "enabled", False) and not _image_has_airguard(image, workspace_root):
        raise RuntimeError(
            f"Container image {image!r} was built without AIRGuard source. "
            "Ensure its Dockerfile honors INSTALL_AIRGUARD=true."
        )
    if _clawsentry_managed(cfg) and not _image_has_clawsentry(image, workspace_root):
        raise RuntimeError(
            f"Container image {image!r} was built without ClawSentry 0.8.7. "
            "Ensure its Dockerfile honors INSTALL_CLAWSENTRY=true."
        )


def _container_name(run_dir: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", run_dir.name).strip("-._").lower()
    slug = slug or "run"
    return f"agent-scaffold-{slug}-{os.getpid()}"[:63].rstrip("-._")


def _prepare_adr_container_config(cfg_path: str, run_dir: Path) -> str:
    raw = load_config_mapping(cfg_path)
    adr = raw.get("adr", {})
    if isinstance(adr, bool):
        adr = {"enabled": adr}
    if not isinstance(adr, dict):
        raise TypeError("adr must be a boolean or mapping")
    adr["detection_root"] = "/opt/adr/Detection"
    adr["python_executable"] = "/opt/adr-venv/bin/python"
    raw["adr"] = adr
    path = run_dir / "adr.container.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return str(path)


def _prepare_airguard_container_config(cfg_path: str, run_dir: Path) -> str:
    raw = load_config_mapping(cfg_path)
    settings = raw.get("airguard", {})
    if isinstance(settings, bool):
        settings = {"enabled": settings}
    if not isinstance(settings, dict):
        raise TypeError("airguard must be a boolean or mapping")
    settings["source_root"] = "/opt/airguard/src"
    raw["airguard"] = settings
    path = run_dir / "airguard.container.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return str(path)


def _container_pid(name: str, workspace_root: Path) -> int:
    inspected = _run_checked(
        ["docker", "inspect", "--format", "{{.State.Pid}}", name],
        workspace_root,
    )
    if inspected.returncode != 0:
        raise RuntimeError(f"Failed to inspect Agent container {name}: {inspected.stderr.strip()}")
    try:
        pid = int(inspected.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Docker returned an invalid PID for {name}: {inspected.stdout!r}") from exc
    if pid <= 0:
        raise RuntimeError(f"Agent container {name} is not running")
    return pid


def _update_container_trace_agentsight(
    result: dict[str, Any],
    agentsight_result: dict[str, Any],
    workspace_root: Path,
    container_workdir: str,
) -> None:
    persist = result.get("_trace_persist")
    if not isinstance(persist, dict) or not persist.get("output_path"):
        return
    container_path = Path(str(persist["output_path"]))
    try:
        relative = container_path.relative_to(Path(container_workdir))
    except ValueError:
        return
    host_path = workspace_root.resolve() / relative
    if not host_path.exists():
        return
    payload = redact_config_snapshot(
        json.loads(host_path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict):
        return
    payload.setdefault("harness", {})["agentsight"] = agentsight_result
    temporary = host_path.with_name(f"{host_path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(host_path)


def run_once_in_container(
    cfg: Any,
    cfg_path: str,
    user_input: str | None,
    context_messages: list[dict[str, str]] | None,
    resume_messages: list[dict[str, str]] | None,
    workspace_root: Path,
    run_dir: Path,
    extra_mounts: list[tuple[str, str]] | None = None,
    image_ready: bool = False,
    run_as_host_user: bool = False,
) -> dict[str, Any]:
    if _clawsentry_managed(cfg):
        endpoint = urlsplit(str(cfg.clawsentry.base_url))
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.path not in {"", "/"}
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
        ):
            raise ValueError(
                "clawsentry.auto_start requires a loopback base_url without a path; "
                "set auto_start: false for an external gateway"
            )
        if not cfg.clawsentry.api_key_env:
            raise ValueError("clawsentry.auto_start requires a nonempty api_key_env")
    if not image_ready:
        _ensure_image(cfg, workspace_root)

    if getattr(getattr(cfg, "adr", None), "enabled", False):
        cfg_path = _prepare_adr_container_config(cfg_path, run_dir)
    if getattr(getattr(cfg, "airguard", None), "enabled", False):
        cfg_path = _prepare_airguard_container_config(cfg_path, run_dir)

    container_workdir = str(cfg.container.workdir).rstrip("/") or "/workspace"
    config_in_container = _workspace_container_path(Path(cfg_path), workspace_root, container_workdir)
    run_dir_in_container = _workspace_container_path(run_dir, workspace_root, container_workdir)

    payload_path = run_dir / "_container_payload.json"
    result_path = run_dir / "_container_result.json"
    stdout_path = run_dir / "container_stdout.log"
    stderr_path = run_dir / "container_stderr.log"
    gate_path = run_dir / "_agentsight_start"
    gate_in_container = f"{run_dir_in_container}/_agentsight_start"
    ready_path = run_dir / "_agentsight_ready"
    ready_in_container = f"{run_dir_in_container}/_agentsight_ready"
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

    name = _container_name(run_dir)
    cmd = ["docker", "run", "-d", "--name", name]
    if getattr(getattr(cfg, "adr", None), "enabled", False):
        run_as_host_user = True
    if run_as_host_user:
        cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"])
    network = str(cfg.container.network or "").strip()
    if network:
        cmd.extend(["--network", network])
    cmd.extend(["-v", f"{workspace_root}:{container_workdir}", "-w", container_workdir])
    for host, target in extra_mounts or []:
        cmd.extend(["--mount", f"type=bind,src={host},dst={target},readonly"])
    cmd.extend(["-e", "PYTHONPATH=src", "-e", "AGENT_CONTAINERIZED=1"])
    cmd.extend(["-e", f"AGENT_JOB_DIR={run_dir_in_container}"])
    cmd.extend(["-e", f"AGENT_RESULT_PATH={run_dir_in_container}/_container_result.json"])
    batch_dir = os.environ.get("AGENT_BATCH_DIR", "").strip()
    if batch_dir:
        batch_dir_in_container = _workspace_container_path(
            Path(batch_dir), workspace_root, container_workdir
        )
        cmd.extend(["-e", f"AGENT_BATCH_DIR={batch_dir_in_container}"])
    if cfg.agentsight.enabled:
        cmd.extend(["-e", "AGENTSIGHT_MANAGED=1"])
        cmd.extend(["-e", f"AGENTSIGHT_START_FILE={gate_in_container}"])
        cmd.extend(["-e", f"AGENTSIGHT_READY_FILE={ready_in_container}"])
        gate_timeout = cfg.agentsight.startup_timeout_seconds + cfg.agentsight.warmup_seconds + 30.0
        cmd.extend(["-e", f"AGENTSIGHT_START_TIMEOUT={gate_timeout}"])
    env_names = [str(name) for name in (getattr(cfg.container, "env", []) or [])]
    llm_api_key_env = str(getattr(cfg.llm, "api_key_env", "") or "")
    if llm_api_key_env and llm_api_key_env not in env_names:
        env_names.append(llm_api_key_env)
    progent_api_key_env = str(
        getattr(getattr(cfg, "progent", None), "api_key_env", "") or ""
    )
    if progent_api_key_env and progent_api_key_env not in env_names:
        env_names.append(progent_api_key_env)
    airguard_api_key_env = str(
        getattr(getattr(cfg, "airguard", None), "api_key_env", "") or ""
    )
    if airguard_api_key_env and airguard_api_key_env not in env_names:
        env_names.append(airguard_api_key_env)
    airguard_generator_key_env = str(
        getattr(getattr(getattr(cfg, "airguard", None), "generator", None), "api_key_env", "") or ""
    )
    if airguard_generator_key_env and airguard_generator_key_env not in env_names:
        env_names.append(airguard_generator_key_env)
    clawsentry_api_key_env = str(
        getattr(getattr(cfg, "clawsentry", None), "api_key_env", "") or ""
    )
    if clawsentry_api_key_env and clawsentry_api_key_env not in env_names:
        env_names.append(clawsentry_api_key_env)
    janus_api_key_env = str(
        getattr(getattr(cfg, "janus", None), "api_key_env", "") or ""
    )
    if janus_api_key_env and janus_api_key_env not in env_names:
        env_names.append(janus_api_key_env)
    stepguard_api_key_env = str(
        getattr(getattr(cfg, "stepguard", None), "api_key_env", "") or ""
    )
    if stepguard_api_key_env and stepguard_api_key_env not in env_names:
        env_names.append(stepguard_api_key_env)
    safeagent_api_key_env = str(
        getattr(getattr(cfg, "safeagent", None), "api_key_env", "") or ""
    )
    if safeagent_api_key_env and safeagent_api_key_env not in env_names:
        env_names.append(safeagent_api_key_env)
    safeagent_generator_key_env = str(
        getattr(getattr(getattr(cfg, "safeagent", None), "generator", None), "api_key_env", "") or ""
    )
    if safeagent_generator_key_env and safeagent_generator_key_env not in env_names:
        env_names.append(safeagent_generator_key_env)
    if getattr(getattr(cfg, "adr", None), "enabled", False):
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
            if name not in env_names:
                env_names.append(name)
    scenario_api_key_env = str(
        getattr(cfg.agentguard.scenario_compiler, "api_key_env", "") or ""
    )
    if scenario_api_key_env and scenario_api_key_env not in env_names:
        env_names.append(scenario_api_key_env)
    for env_name in env_names:
        if env_name in os.environ:
            # Let Docker copy the value from its own environment. Passing only
            # the name keeps credentials out of the process argument list.
            cmd.extend(["-e", env_name])
    if _clawsentry_managed(cfg):
        cmd.extend(["-e", f"AGENT_CLAWSENTRY_KEY_ENV={cfg.clawsentry.api_key_env}"])
    cmd.append(str(cfg.container.image))
    agent_command = [
            sys.executable.split("/")[-1] if sys.executable else "python",
            "src/agent_scaffold/main.py",
            "--config",
            config_in_container,
            "--run-payload",
            f"{run_dir_in_container}/_container_payload.json",
    ]
    if _clawsentry_managed(cfg):
        cmd.extend(["python", "-m", "agent_scaffold.clawsentry.launcher", "--", *agent_command])
    else:
        cmd.extend(agent_command)

    observer: AgentSightObserver | None = None
    observer_stopped = False
    container_started = False
    container_completed = False
    try:
        started = _run_checked(cmd, workspace_root)
        if started.returncode != 0:
            raise RuntimeError(
                f"Failed to start agent container {name}.\n"
                f"Command: {' '.join(cmd)}\nSTDOUT:\n{started.stdout}\nSTDERR:\n{started.stderr}"
            )
        container_started = True

        if cfg.agentsight.enabled:
            ready_deadline = time.monotonic() + cfg.agentsight.startup_timeout_seconds
            while not ready_path.exists():
                if time.monotonic() >= ready_deadline:
                    raise RuntimeError(
                        f"Agent container {name} did not reach the AgentSight ready gate "
                        f"within {cfg.agentsight.startup_timeout_seconds}s"
                    )
                time.sleep(0.05)
            observer = AgentSightObserver(
                cfg.agentsight,
                run_dir,
                target_pid=_container_pid(name, workspace_root),
                binary_path=f"docker://{name}",
            )
            observer.start()
            gate_path.touch()

        waited = _run_checked(["docker", "wait", name], workspace_root)
        logs = _run_checked(["docker", "logs", name], workspace_root)
        stdout_path.write_text(logs.stdout, encoding="utf-8")
        stderr_path.write_text(logs.stderr, encoding="utf-8")

        agentsight_result: dict[str, Any] | None = None
        if observer is not None:
            try:
                agentsight_result = observer.stop()
            finally:
                observer_stopped = True

        try:
            container_returncode = int(waited.stdout.strip()) if waited.returncode == 0 else 1
        except ValueError:
            container_returncode = 1
        if waited.returncode != 0 or container_returncode != 0:
            error = {
                "error": "container_run_failed",
                "returncode": container_returncode,
                "command": cmd,
                "container_name": name,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_tail": logs.stdout[-4000:],
                "stderr_tail": logs.stderr[-4000:],
                "agentsight": agentsight_result,
            }
            (run_dir / "container_error.json").write_text(
                json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                f"Container failed with exit code {container_returncode}. "
                f"See {stderr_path} and {stdout_path}."
            )

        if not result_path.exists():
            raise RuntimeError(f"Container completed but did not write result file: {result_path}")
        result = redact_config_snapshot(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
        result_temporary = result_path.with_name(f"{result_path.name}.tmp")
        result_temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        result_temporary.replace(result_path)
        if agentsight_result is not None:
            result.setdefault("harness", {})["agentsight"] = agentsight_result
            _update_container_trace_agentsight(
                result, agentsight_result, workspace_root, container_workdir
            )
            result_temporary = result_path.with_name(f"{result_path.name}.tmp")
            result_temporary.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            result_temporary.replace(result_path)
        container_completed = True
        return result
    finally:
        if cfg.agentsight.enabled and not gate_path.exists():
            gate_path.touch()
        if observer is not None and not observer_stopped:
            try:
                observer.stop()
            except Exception:
                if cfg.agentsight.required:
                    raise
        if container_started and (bool(cfg.container.remove) or not container_completed):
            _run_checked(["docker", "rm", "-f", name], workspace_root)
