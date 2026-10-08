#!/usr/bin/env python3
"""Replay selected Hermes guards on recorded lifecycles; never rerun Hermes.

Successful results live under each phase's defense_supplements directory. An
atomic defense_supplements.json overlay selects them for the three analyzers;
original defense_replay outputs and benchmark evaluations stay untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_monitor_rules import METHODS, SUPPLEMENT_MANIFEST, label, replay_sources

REPO = Path(__file__).resolve().parents[1]
METHOD_SET = set(METHODS)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _workspace_path(path: Path) -> str:
    return "/workspace/" + path.resolve().relative_to(REPO).as_posix()


def _resolve_batches(path: Path, batch_name: str) -> list[Path]:
    path = path.resolve()
    if path.name == "all_monitors":
        on = path / "skill_injection"
        off = path / "no_injection"
        names = {p.parent.name for p in on.glob("*/summary.json")}
        names &= {p.parent.name for p in off.glob("*/summary.json")}
        if batch_name:
            if batch_name not in names:
                raise ValueError(f"No completed paired batch named {batch_name} under {path}")
            selected = batch_name
        else:
            if not names:
                raise ValueError(f"No completed paired batch under {path}")
            selected = sorted(names)[-1]
        return [on / selected, off / selected]
    if batch_name:
        raise ValueError("--batch only applies to an AgentDojo all_monitors directory")
    if not (path / "summary.json").is_file():
        raise ValueError(f"Batch has no summary.json: {path}")
    return [path]


def _benchmark(batch: Path) -> str:
    config = (_json(batch / "summary.json").get("config") or {}).get("agent.yaml") or {}
    if (config.get("agentdojo") or {}).get("enabled"):
        return "agentdojo"
    if (config.get("agent_security_bench") or {}).get("enabled"):
        return "asb"
    if (config.get("privacylens_live") or {}).get("enabled"):
        return "privacylens_live"
    raise ValueError(f"Unsupported Hermes benchmark in {batch / 'summary.json'}")


def _phases(batch: Path, benchmark: str) -> list[tuple[Path, Path]]:
    summary = _json(batch / "summary.json")
    names = ("target", "control") if benchmark == "asb" else ("target",)
    pairs = []
    for item in sorted(summary.get("items") or [], key=lambda x: x["index"]):
        run = batch / f"run_{item['index']:03d}"
        for name in names:
            phase = run / name
            evaluation = run / "result.json" if benchmark == "agentdojo" else phase / "evaluation.json"
            if all(path.is_file() for path in (phase / "guard_lifecycle.jsonl", phase / "defenses.json", evaluation)):
                pairs.append((run, phase))
            else:
                print(f"skip incomplete phase: {phase}", file=sys.stderr)
    return pairs


def _existing_label(phase: Path, manifest: dict, method: str) -> dict | None:
    sources = replay_sources(phase, manifest)
    if method not in sources:
        return None
    source = sources[method]
    status = source["status"]
    error = "" if status.get("status") == "completed" else str(
        status.get("error") or status.get("status") or "missing replay status"
    )
    try:
        defense = _json(source["path"])
    except (FileNotFoundError, json.JSONDecodeError):
        defense = None
        error = error or "missing or invalid defense replay"
    return label(method, defense, error)


def _selected_env(config: dict) -> list[str]:
    names = set((config.get("container") or {}).get("env") or [])
    names.update(("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "TOGETHER_API_KEY"))

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_env") and isinstance(item, str) and item:
                    names.add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config)
    return sorted(name for name in names if isinstance(name, str) and name in os.environ and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _image(config: dict, override: str) -> str:
    if override:
        return override
    base = str((config.get("container") or {}).get("image") or "")
    if not base:
        raise ValueError("Saved config has no container.image; supply --image")
    result = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        text=True, capture_output=True, check=True,
    )
    candidates = [line for line in result.stdout.splitlines() if line.startswith(base + "-base-")]
    if not candidates:
        raise ValueError(f"No local replay image for {base}; supply --image")
    return candidates[0]  # docker image ls is newest first; --image pins an exact tag.


_REMOVED_PRO2GUARD_FIELDS = (
    "unsafe_states", "horizon", "dtmc_path", "abstraction", "abstraction_policy_path",
)


_PATH_KEYS = {"path", "policy", "plugin_config", "model_path", "dtmc_path"}
# Values the batch runner sets per batch (backends/container.py
# STATE_PATH_KEYS) or per image; an override keeps the saved ones.
_SAVED_KEYS = ("memory_path", "router_cache_path", "source_root", "detection_root", "python_executable")


def _method_sections(path: Path, methods: list[str]) -> dict[str, dict]:
    """Selected methods' sections from a YAML, with paths mapped as in the container.

    Relative path values resolve against the YAML's directory, then the
    repository root, as backends/container.py does when it writes
    hermes.container.yaml.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent

    def remap(value, key=""):
        if isinstance(value, dict):
            return {k: remap(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v, key) for v in value]
        if not (isinstance(value, str) and value and (key in _PATH_KEYS or key.endswith(("_path", "_dir", "_file")))):
            return value
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / value if (base / value).exists() else REPO / value
        if not candidate.exists():
            return value
        try:
            return _workspace_path(candidate)
        except ValueError:
            raise ValueError(f"{path}: {key}={value!r} is outside the workspace and is not mounted in replays")

    sections = {}
    for method in methods:
        section = raw.get(method)
        if not isinstance(section, dict):
            raise ValueError(f"{path} has no {method}: section")
        sections[method] = remap(section)
    return sections


def _replay_config(config_path: Path, attempt: Path, config: dict, force: bool = False) -> tuple[Path, list[str]]:
    """Load old saved configs without changing the benchmark's original YAML."""
    pro2guard = config.get("pro2guard") or {}
    removed = [name for name in _REMOVED_PRO2GUARD_FIELDS if name in pro2guard]
    if not removed and not force:
        return config_path, []
    updated = dict(config)
    updated["pro2guard"] = {key: value for key, value in pro2guard.items() if key not in removed}
    harness = updated.get("harness")
    if isinstance(harness, str) and harness and not Path(harness).is_absolute():
        updated["harness"] = str((config_path.parent / harness).resolve())
    elif isinstance(harness, dict):
        harness = dict(harness)
        for key in ("path", "dir"):
            if harness.get(key) and not Path(str(harness[key])).is_absolute():
                harness[key] = str((config_path.parent / str(harness[key])).resolve())
        updated["harness"] = harness
    path = attempt / "hermes.replay.yaml"
    path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    return path, removed


def _worker(config_path: Path, phase: Path, method: str, attempt: Path) -> int:
    from agent_scaffold.backends.guards import GUARDS, replay_guards
    from agent_scaffold.config import load_config

    cfg = load_config(config_path)
    # Generators cache at batch scope. Seed a private copy to preserve cache hits
    # without letting a supplemental run change the original experiment state.
    for source in phase.parent.parent.glob("*.batch-cache.json"):
        shutil.copy2(source, attempt / source.name)
    os.environ["AGENT_POLICY_CACHE_DIR"] = str(attempt)
    if method == "agrail" and cfg.agrail.memory_path:
        source = Path(cfg.agrail.memory_path)
        isolated = attempt / "agrail-memory.json"
        isolated.write_bytes(source.read_bytes() if source.is_file() else b"[]")
        cfg.agrail.memory_path = str(isolated)
    for name in GUARDS:
        getattr(cfg, name).enabled = name == method
    manifest = replay_guards(cfg, phase / "guard_lifecycle.jsonl", attempt)
    status = manifest.get("methods", {}).get(method, {})
    print(json.dumps({"method": method, "status": status.get("status"), "error": status.get("error", "")}, ensure_ascii=False))
    return 0 if status.get("status") == "completed" else 1


def _replay(run: Path, phase: Path, method: str, attempt: Path, image: str, timeout: int, config: dict, replay_config: Path) -> tuple[int, str]:
    name = "hermes-supplement-" + uuid.uuid4().hex[:12]
    command = [
        "docker", "run", "--rm", "--name", name, "--network", "host",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
        "-v", f"{REPO}:/workspace", "-w", "/workspace",
        "-e", "PYTHONPATH=src", "-e", "AGENT_CONTAINERIZED=1",
        "-e", f"AGENT_BATCH_DIR={_workspace_path(run.parent)}",
        "-e", f"AGENT_JOB_DIR={_workspace_path(run)}",
    ]
    for env_name in _selected_env(config):
        command.extend(["-e", env_name])
    managed_clawsentry = method == "clawsentry" and (config.get("clawsentry") or {}).get("auto_start", True)
    if managed_clawsentry:
        command.extend(["-e", "AGENT_CLAWSENTRY_KEY_ENV=CS_AUTH_TOKEN"])
    command.append(image)
    worker = [
        "python", "scripts/supplement_hermes_monitors.py", "--worker",
        "--config", _workspace_path(replay_config),
        "--phase", _workspace_path(phase), "--method", method,
        "--attempt", _workspace_path(attempt),
    ]
    if managed_clawsentry:
        command.extend(["python", "-m", "agent_scaffold.clawsentry.launcher", "--", *worker])
    else:
        command.extend(worker)
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        (attempt / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (attempt / "stderr.log").write_text(result.stderr, encoding="utf-8")
        return result.returncode, result.stderr[-1000:]
    except subprocess.TimeoutExpired as exc:
        for filename, content in (("stdout.log", exc.stdout), ("stderr.log", exc.stderr)):
            if content is not None:
                (attempt / filename).write_bytes(
                    content if isinstance(content, bytes) else content.encode("utf-8")
                )
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise
    except KeyboardInterrupt:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise


def _activate(phase: Path, method: str, attempt: Path, result_path: Path, image: str) -> None:
    path = phase / SUPPLEMENT_MANIFEST
    with (phase / ".defense_supplements.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        overlay = _json(path) if path.exists() else {"version": 1, "methods": {}}
        if overlay.get("version") != 1:
            raise ValueError(f"Unsupported supplement manifest: {path}")
        overlay.setdefault("methods", {})[method] = {
            "path": str(result_path.relative_to(phase)),
            "attempt": str(attempt.relative_to(phase)),
            "image": image,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(path, overlay)


def _analyze(batch: Path, benchmark: str, attempt_id: str) -> None:
    if benchmark == "agentdojo":
        root = batch.parent.parent.parent
        script = REPO / "scripts/analyze_hermes_agentdojo_monitors.py"
        command = [sys.executable, str(script), "--root", str(root), "--batch", batch.name]
    elif benchmark == "asb":
        script = REPO / "scripts/analyze_hermes_asb_monitors.py"
        command = [sys.executable, str(script), str(batch)]
        root = batch
    else:
        script = REPO / "scripts/analyze_hermes_privacylens_live_monitors.py"
        command = [sys.executable, str(script), str(batch)]
        root = batch
    analysis = root / "analysis"
    if analysis.exists():
        backup = analysis / "before_supplement" / attempt_id
        backup.mkdir(parents=True, exist_ok=True)
        for path in analysis.iterdir():
            if path.is_file():
                shutil.copy2(path, backup / path.name)
    subprocess.run(command, cwd=REPO, check=True)
    print(f"updated analysis: {analysis}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", help="batch directory, or AgentDojo all_monitors directory")
    parser.add_argument("--batch", default="", help="exact paired batch name when path is AgentDojo all_monitors")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="guards to replay")
    parser.add_argument("--all-runs", action="store_true", help="replay every completed run; default retries error labels only")
    parser.add_argument("--run", type=int, action="append", help="limit to a run index (repeatable)")
    parser.add_argument("--image", default="", help="exact prebuilt Docker image; default newest matching base image")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per method and phase (default: 1800)")
    parser.add_argument("--dry-run", action="store_true", help="print planned replays without Docker or writes")
    parser.add_argument(
        "--method-config", type=Path, default=None,
        help="YAML whose sections replace each selected method's saved config "
             "(e.g. the current agents/hermes config); other settings stay as saved",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--phase", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not all((args.config, args.phase, args.method, args.attempt)):
            parser.error("worker requires --config, --phase, --method and --attempt")
        return _worker(args.config, args.phase, args.method, args.attempt)
    if not args.path or not args.methods:
        parser.error("path and --methods are required")
    if args.timeout < 1 or any(index < 1 for index in args.run or []):
        parser.error("--timeout and --run must be positive")
    overrides = _method_sections(args.method_config, list(dict.fromkeys(args.methods))) if args.method_config else {}
    batches = _resolve_batches(args.path, args.batch)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    selected_runs = set(args.run or [])
    planned: list[tuple[Path, str, Path, Path, str]] = []
    kinds: dict[Path, str] = {}
    for batch in batches:
        if not batch.is_relative_to(REPO):
            raise ValueError(f"Batch must be inside the workspace: {batch}")
        kind = _benchmark(batch)
        kinds[batch] = kind
        for run, phase in _phases(batch, kind):
            if selected_runs and int(run.name.removeprefix("run_")) not in selected_runs:
                continue
            manifest = _json(phase / "defenses.json")
            if manifest.get("mode") != "replay":
                print(f"skip non-replay phase: {phase}", file=sys.stderr)
                continue
            config = run / "hermes.container.yaml"
            if not config.is_file():
                print(f"skip missing saved config: {config}", file=sys.stderr)
                continue
            for method in dict.fromkeys(args.methods):
                old = _existing_label(phase, manifest, method)
                if old is None:
                    print(f"skip method absent from original replay: {phase} {method}", file=sys.stderr)
                elif args.all_runs or old["status"] == "error":
                    planned.append((batch, kind, run, phase, method))
    if not planned:
        print("No matching replay errors. Use --all-runs to recompute valid labels.")
        return 0
    print(f"planned supplemental replays: {len(planned)}")
    for _, _, run, phase, method in planned:
        print(f"  {run.parent.parent.name}/{run.parent.name}/{run.name}/{phase.name}: {method}")
    if args.dry_run:
        return 0
    activated: set[Path] = set()
    succeeded = failed = 0
    for batch, kind, run, phase, method in planned:
        config_path = run / "hermes.container.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if method in overrides:
            saved = config.get(method) if isinstance(config.get(method), dict) else {}
            section = {**overrides[method], **{k: saved[k] for k in _SAVED_KEYS if k in saved}}
            config = {**config, method: section}
        image = _image(config, args.image)
        attempt = phase / "defense_supplements" / attempt_id / method
        attempt.mkdir(parents=True, exist_ok=False)
        replay_config, removed_fields = _replay_config(
            config_path, attempt, config, force=method in overrides
        )
        print(f"replaying {batch.name}/{run.name}/{phase.name}/{method} using {image}", flush=True)
        try:
            code, diagnostic = _replay(run, phase, method, attempt, image, args.timeout, config, replay_config)
        except subprocess.TimeoutExpired:
            code, diagnostic = 124, f"replay exceeded {args.timeout} seconds"
        output = attempt / "defense_replay" / method / "defenses.json"
        method_manifest = attempt / "defenses.json"
        status = {}
        if method_manifest.exists():
            status = (_json(method_manifest).get("methods") or {}).get(method) or {}
        try:
            defense = _json(output)
        except (FileNotFoundError, json.JSONDecodeError):
            defense = None
        replay_error = "" if code == 0 and status.get("status") == "completed" else (
            str(status.get("error") or diagnostic or f"container exit {code}")
        )
        result_label = label(method, defense, replay_error)
        accepted = result_label["status"] != "error" and result_label["decision_records"] > 0
        record = {
            "method": method, "batch": str(batch), "phase": str(phase),
            "image": image, "container_exit_code": code,
            "replay_status": status.get("status", "missing"),
            "label": result_label, "activated": accepted,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "replay_config_sha256": hashlib.sha256(replay_config.read_bytes()).hexdigest(),
            "removed_legacy_fields": [f"pro2guard.{name}" for name in removed_fields],
            "method_config": (
                {"path": str(args.method_config), "section": config[method]}
                if method in overrides else None
            ),
            "lifecycle_sha256": hashlib.sha256((phase / "guard_lifecycle.jsonl").read_bytes()).hexdigest(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(attempt / "attempt.json", record)
        if accepted:
            _activate(phase, method, attempt, output, image)
            activated.add(batch)
            succeeded += 1
        else:
            failed += 1
            print(f"supplement not activated: {method} {phase} ({result_label['error'] or diagnostic or 'no decisions'})", file=sys.stderr)
    # AgentDojo conditions share one analysis directory; run it once per batch.
    analyzed: set[tuple[str, str]] = set()
    for batch in sorted(activated):
        key = (kinds[batch], batch.name) if kinds[batch] == "agentdojo" else (kinds[batch], str(batch))
        if key not in analyzed:
            _analyze(batch, kinds[batch], attempt_id)
            analyzed.add(key)
    print(f"activated {succeeded} supplemental replays; {failed} remained invalid")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
