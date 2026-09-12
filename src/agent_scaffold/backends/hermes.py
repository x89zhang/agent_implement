"""Hermes backend orchestration. No project agent graph is used."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

from ..agentdojo_adapter import config_file_snapshot, redact_config_snapshot
from .benchmark import BenchmarkService
from .guards import GUARDS
from .memory import copy_memory, manifest
from .skills import stage_skills


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            redact_config_snapshot(value), ensure_ascii=False, indent=2, default=str
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def print_evaluation(name: str, evaluation: dict) -> None:
    """Match the builtin runner concise benchmark evaluation output."""
    print(
        f"{name} evaluation: "
        f"utility={evaluation.get('utility')} "
        f"security={evaluation.get('security')} "
        f"attack_success={evaluation.get('attack_success')} "
        f"score={evaluation.get('score')}"
    )


def print_result_evaluation(result: dict) -> None:
    """Print the benchmark evaluation carried by a completed Hermes result."""
    harness = result.get("harness", {})
    if not isinstance(harness, dict):
        return
    for name in ("agentdojo", "agentharm", "agent_security_bench"):
        evaluation = harness.get(name)
        if isinstance(evaluation, dict):
            print_evaluation(name, evaluation)
            return


def preflight(cfg):
    config = cfg.execution.hermes
    if any(
        importlib.util.find_spec(name) is None
        for name in ("mcp", "jsonschema", "psutil")
    ):
        raise ValueError(
            "Install requirements-hermes-bridge.txt in the project interpreter"
        )
    if cfg.execution.memory.mode != "off" and not cfg.agent_security_bench.enabled:
        raise ValueError("Native memory experiments currently require ASB")
    if cfg.llm.provider not in {"openai", "openrouter"}:
        raise ValueError(
            "Hermes backend currently supports OpenAI-compatible endpoints (provider openai or openrouter)"
        )
    repo = Path(config.repo_path)
    if not (repo / "run_agent.py").is_file():
        raise ValueError(f"Hermes checkout not found: {repo}")
    # Upstream loads repo/.env at import time; disallow accidental profile/credential contamination.
    if (repo / ".env").exists():
        raise ValueError(
            "Use a Hermes checkout without a repository .env; pass model credentials through the configured environment variable"
        )
    interpreter = (
        Path(config.python_executable)
        if config.python_executable
        else repo / ".venv/bin/python"
    )
    if not interpreter.is_file():
        raise ValueError(
            f"Hermes interpreter not found: {interpreter}; configure execution.hermes.python_executable"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    if config.expected_commit and config.expected_commit != commit:
        raise ValueError(
            f"Hermes revision mismatch: expected {config.expected_commit}, found {commit}"
        )
    if dirty and not config.allow_dirty_checkout:
        raise ValueError(
            "Hermes checkout has local changes; explicitly set allow_dirty_checkout to run it"
        )
    metadata = {
        "backend": "hermes",
        "repo_path": str(repo),
        "commit": commit,
        "dirty": bool(dirty),
        "python_executable": str(interpreter),
        "memory_protocol": cfg.execution.memory.mode,
        "guards": [name for name in GUARDS if getattr(cfg, name).enabled],
        "os_sandbox": "docker"
        if os.environ.get("AGENT_CONTAINERIZED") == "1"
        else "external-or-none",
    }
    if dirty:
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--binary"]
        )
        metadata["tracked_patch_sha256"] = hashlib.sha256(diff).hexdigest()
        metadata["working_tree_status"] = dirty
    return interpreter, metadata


def _terminate(process, descendants):
    import psutil

    # MCP stdio children use start_new_session=True, so retain process identities
    # and stop them explicitly as well as the worker's own process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(descendants), timeout=1)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=1)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _run_worker(command, env, workspace, phase_dir, timeout):
    import psutil

    with (
        (phase_dir / "worker.stdout.log").open("w") as stdout,
        (phase_dir / "worker.stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        descendants = set()
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    descendants.update(
                        psutil.Process(process.pid).children(recursive=True)
                    )
                except psutil.NoSuchProcess:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Hermes phase exceeded {timeout} seconds")
                try:
                    return process.wait(timeout=min(remaining, 0.1))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            _terminate(process, descendants)


def _run_phase(
    cfg, interpreter, directory: Path, memory_source: Path | None, prompt_override=None
):
    directory.mkdir(parents=True, exist_ok=False)
    home, workspace = directory / "home", directory / "workspace"
    home.mkdir(mode=0o700)
    workspace.mkdir()
    skills = stage_skills(cfg, home)
    dump(directory / "skills.json", skills)
    copy_memory(memory_source, home)
    initial_memory = manifest(home)
    enabled_memory = cfg.execution.memory.mode != "off"
    dump(directory / "memory.before.json", initial_memory)
    source_root = Path(__file__).resolve().parents[2]
    project_root = source_root.parent
    result_path = directory / "worker_result.json"
    settings = cfg.execution.hermes
    with BenchmarkService(
        cfg, directory / "tool_calls.jsonl", settings.timeout_seconds
    ) as bridge:
        name, tools = bridge.name, bridge.tools
        prompt = bridge.task if prompt_override is None else prompt_override
        hermes_config = {
            "tools": {"tool_search": {"enabled": "off"}},
            "memory": {
                "memory_enabled": enabled_memory,
                "user_profile_enabled": enabled_memory,
            },
            "mcp_servers": {
                "benchmark": {
                    "command": sys.executable,
                    "args": ["-m", "agent_scaffold.backends.mcp_proxy"],
                    "env": {
                        "PYTHONPATH": str(source_root),
                        "BENCHMARK_BRIDGE_URL": bridge.url,
                        "BENCHMARK_BRIDGE_TOKEN": "${BENCHMARK_BRIDGE_TOKEN}",
                        "BENCHMARK_BRIDGE_TIMEOUT": str(settings.timeout_seconds),
                    },
                    "timeout": settings.timeout_seconds,
                }
            },
        }
        (home / "config.yaml").write_text(
            yaml.safe_dump(hermes_config, sort_keys=False)
        )
        request = {
            "schema_version": 1,
            "repo_path": settings.repo_path,
            "prompt": prompt,
            "model": cfg.llm.model,
            "provider": "custom" if cfg.llm.provider == "openai" else "openrouter",
            "base_url": cfg.llm.base_url
            or (
                "https://api.openai.com/v1"
                if cfg.llm.provider == "openai"
                else "https://openrouter.ai/api/v1"
            ),
            "api_mode": settings.api_mode,
            "temperature": cfg.llm.temperature,
            "max_iterations": settings.max_iterations,
            "timeout_seconds": settings.timeout_seconds,
            "memory_enabled": enabled_memory,
            "skills": skills,
            "guard_url": bridge.url,
            "tool_count": len(tools),
            "tool_names": [tool["name"] for tool in tools],
        }
        dump(directory / "request.json", request)
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "LANG",
                "LC_ALL",
                "TMPDIR",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "REQUESTS_CA_BUNDLE",
            }
        }
        # Preserve the real HOME variable; use the upstream profile mechanism for state isolation.
        env.update(
            {
                "HERMES_HOME": str(home),
                "BENCHMARK_BRIDGE_TOKEN": bridge.token,
                "BENCHMARK_GUARD_TOKEN": bridge.guard_token,
                "BENCHMARK_MODEL_API_KEY": cfg.llm.api_key
                or os.environ.get(cfg.llm.api_key_env, ""),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if "HOME" in os.environ:
            env["HOME"] = os.environ["HOME"]
        command = [
            str(interpreter),
            str(project_root / "integrations/hermes/worker.py"),
            str(directory / "request.json"),
            str(result_path),
        ]
        try:
            code = _run_worker(
                command, env, workspace, directory, settings.timeout_seconds
            )
            if not result_path.exists():
                raise RuntimeError(
                    f"Hermes exited {code} without a result; see {directory / 'worker.stderr.log'}"
                )
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict) or result.get("schema_version") != 1:
                raise RuntimeError("Unsupported Hermes worker result schema")
            if code != 0 or result.get("status") != "completed":
                raise RuntimeError(
                    f"Hermes failed: {result.get('termination_reason')}; see {directory}"
                )
        except Exception as exc:
            dump(
                directory / "failure.json",
                {
                    "status": "timeout" if isinstance(exc, TimeoutError) else "failed",
                    "error": str(exc),
                },
            )
            raise
        try:
            evaluation = bridge.evaluate(result["final_output"])
        except Exception as exc:
            dump(
                directory / "failure.json",
                {"status": "evaluation_failed", "error": str(exc)},
            )
            raise
    events_path = directory / "events.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    for skill in skills:
        reads = [
            event["payload"]
            for event in events
            if event["type"] == "skill_view"
            and event["payload"].get("arguments", {}).get("name") == skill["name"]
        ]
        skill["read_attempted"] = bool(reads) or any(
            event["type"] == "skill_view_attempt"
            and event["payload"].get("arguments", {}).get("name") == skill["name"]
            for event in events
        )
        skill["original_content_delivered"] = any(
            not r["failed"]
            and r["result_allowed"]
            and not r.get("result_changed", False)
            for r in reads
        )
    dump(directory / "skills.exposure.json", skills)
    dump(directory / "evaluation.json", evaluation)
    dump(directory / "memory.after.json", manifest(home))
    return {
        "benchmark": name,
        "evaluation": evaluation,
        "worker": result,
        "memory_before": initial_memory,
        "memory_after": manifest(home),
        "home": home,
        "calls": bridge.calls,
        "prompt": prompt,
        "defenses": json.loads((directory / "defenses.json").read_text()),
        "skills": skills,
    }


def run_phase(cfg, interpreter, directory, memory_source, prompt_override=None):
    try:
        return _run_phase(cfg, interpreter, directory, memory_source, prompt_override)
    except Exception as exc:
        failure = Path(directory) / "failure.json"
        if not failure.exists():
            dump(
                failure,
                {
                    "status": "timeout" if isinstance(exc, TimeoutError) else "failed",
                    "error": str(exc),
                },
            )
        raise


def _run_hermes(
    cfg, cfg_path, run_dir, user_input=None, context_messages=None, resume_messages=None
):
    if context_messages or resume_messages:
        raise ValueError(
            "Hermes does not accept project conversation replay; each phase starts a new upstream session"
        )
    interpreter, metadata = preflight(cfg)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dump(run_dir / "config.snapshot.json", config_file_snapshot(cfg_path))
    dump(run_dir / "hermes.metadata.json", metadata)
    memory = cfg.execution.memory
    clean = (
        Path(memory.clean_initial_memory_dir)
        if memory.clean_initial_memory_dir
        else None
    )
    target_cfg = copy.deepcopy(cfg)
    target_cfg.agent.task = "\n\n".join(
        part for part in (cfg.agent.task, user_input) if part
    )
    source = clean
    lifecycle = {
        "mode": memory.mode,
        "retained_surfaces": ["MEMORY.md", "USER.md"],
        "session_history_retained": False,
        "semantic_poisoning_verified": None,
    }
    if memory.mode != "off":
        # Suppress legacy synthetic memory injection in BOTH phase environments and the target task.
        target_cfg.agent_security_bench = replace(
            target_cfg.agent_security_bench, injection_method="memory_attack"
        )
        if memory.mode == "native_two_stage":
            poison_text = (
                Path(memory.poisoning_input_file).read_text(encoding="utf-8").strip()
            )
            if not poison_text:
                raise ValueError("poisoning_input_file must not be empty")
            poison = run_phase(
                target_cfg, interpreter, run_dir / "poison", clean, poison_text
            )
            source = poison["home"] / "memories"
            lifecycle["memory_changed"] = (
                poison["memory_before"] != poison["memory_after"]
            )
            lifecycle["carrier"] = "user_message"
        else:
            source = Path(memory.poisoned_memory_dir)
            lifecycle["carrier"] = "controller_direct_seed"
    target = run_phase(target_cfg, interpreter, run_dir / "target", source)
    lifecycle["target_attack_success"] = target["evaluation"].get("attack_success")
    lifecycle["target_utility"] = target["evaluation"].get("utility")
    if memory.mode != "off" and memory.run_clean_control:
        control = run_phase(target_cfg, interpreter, run_dir / "control", clean)
        lifecycle["control_evaluation"] = control["evaluation"]
    dump(run_dir / "memory.lifecycle.json", lifecycle)
    worker = target["worker"]
    evaluation = target["evaluation"]
    if memory.mode != "off":
        evaluation = {
            **evaluation,
            "evaluation_protocol": "asb_derived_native_memory",
            "memory_mode": memory.mode,
            "original_injection_method": cfg.agent_security_bench.injection_method,
        }
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "messages": [
            {"role": "user", "content": target["prompt"]},
            {"role": "assistant", "content": worker["final_output"]},
        ],
        "trace_messages": worker["messages"],
        "trace": target["defenses"]["trace"]
        + target["calls"]
        + [{"step": target["benchmark"] + "_eval", "output": evaluation}],
        "trace_stats": {"api_calls": worker.get("api_calls")},
        "harness": {
            **target["defenses"]["harness"],
            "defenses": target["defenses"],
            "skills": target["skills"],
            target["benchmark"]: evaluation,
            "hermes": metadata,
            "memory_experiment": lifecycle,
        },
    }
    if cfg.monitoring.print_trace:
        print_evaluation(target["benchmark"], evaluation)
    dump(run_dir / "result.json", result)
    output = Path(cfg.monitoring.output_path or f"trace_{target['benchmark']}.json")
    if not output.is_absolute():
        output = run_dir / output
    dump(output, result)
    return result


def run_hermes(
    cfg, cfg_path, run_dir, user_input=None, context_messages=None, resume_messages=None
):
    """Attach the existing observer lifecycle to the external process tree."""
    from ..agentsight import AgentSightObserver

    observer = None
    result = None
    if cfg.agentsight.enabled and os.environ.get("AGENTSIGHT_MANAGED") != "1":
        observer = AgentSightObserver(
            cfg.agentsight, Path(run_dir), target_pid=os.getpid()
        )
        observer.start()
    try:
        result = _run_hermes(
            cfg, cfg_path, run_dir, user_input, context_messages, resume_messages
        )
        if cfg.agentsight.enabled:
            result["harness"]["agentsight"] = {
                "enabled": True,
                "status": "managed_by_host",
            }
        output = Path(
            cfg.monitoring.output_path
            or f"trace_{next(n for n in ('agentdojo', 'agentharm', 'agent_security_bench') if getattr(cfg, n).enabled)}.json"
        )
        if not output.is_absolute():
            output = Path(run_dir) / output
        result["_trace_persist"] = {"output_path": str(output)}
        return result
    finally:
        if observer is not None:
            observed = observer.stop()
            if result is not None:
                result["harness"]["agentsight"] = observed
        if result is not None:
            dump(Path(run_dir) / "result.json", result)
            dump(Path(result["_trace_persist"]["output_path"]), result)
