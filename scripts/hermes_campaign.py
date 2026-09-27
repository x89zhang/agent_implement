#!/usr/bin/env python3
"""Shared campaign setup for the Hermes all-monitors run scripts.

Every benchmark is evaluated with the guard set of the AgentDojo all-monitors
reference YAML: each guard enabled there is enabled with the same (passive)
mode, every other guard is disabled. Guard-specific settings stay those of the
benchmark YAML.

Each campaign also gets a fresh AGrail memory file under its batch directory
(upstream starts every dataset run with a new memory file); its path and hashes
are recorded in ``campaign_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import yaml

from hermes_monitor_rules import METHODS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "agents/hermes/agentdojo-all-monitors.yaml"
MANIFEST = "campaign_manifest.json"
AGRAIL_MEMORY = "agrail-memory.json"


def reference_guards(reference: pathlib.Path = REFERENCE) -> dict[str, dict]:
    raw = yaml.safe_load(reference.read_text(encoding="utf-8")) or {}
    guards = {}
    for name in METHODS:
        section = raw.get(name) or {}
        enabled = bool(section.get("enabled"))
        guards[name] = {"enabled": enabled}
        if enabled and section.get("mode"):
            guards[name]["mode"] = section["mode"]
    return guards


def apply_reference_guards(raw: dict, reference: pathlib.Path = REFERENCE) -> list[str]:
    """Enable exactly the reference guard set on ``raw``; return enabled names."""
    guards = reference_guards(reference)
    for name, values in guards.items():
        raw.setdefault(name, {}).update(values)
    return [name for name, values in guards.items() if values["enabled"]]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_agrail_memory(raw: dict, directory: pathlib.Path) -> dict | None:
    """Point AGrail at a new, empty campaign memory file under ``directory``."""
    if not (raw.get("agrail") or {}).get("enabled"):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / AGRAIL_MEMORY).resolve()
    if path.exists():
        raise SystemExit(f"AGrail campaign memory already exists: {path}")
    path.write_text("[]", encoding="utf-8")
    raw["agrail"]["memory_path"] = str(path)
    return {"path": str(path), "sha256_initial": _sha256(path)}


def campaign_memory(batch_dir: pathlib.Path) -> pathlib.Path:
    """Create (or, when a campaign resumes, keep) its AGrail memory file."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = (batch_dir / AGRAIL_MEMORY).resolve()
    manifest = batch_dir / MANIFEST
    recorded = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
        record_manifest(batch_dir, "campaign", {
            **(recorded.get("campaign") or {}),
            "agrail_memory": {"path": str(path), "sha256_initial": _sha256(path)},
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return path


def record_manifest(batch_dir: pathlib.Path, key: str, value: dict) -> None:
    path = batch_dir / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest[key] = value
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def configure(
    raw: dict, batch_dir: pathlib.Path, label: str = "campaign",
    memory_dir: pathlib.Path | None = None, source: str = "",
    reference: pathlib.Path = REFERENCE,
) -> dict:
    """Apply the reference guard set and a fresh AGrail memory; record both."""
    enabled = apply_reference_guards(raw, reference)
    memory = fresh_agrail_memory(raw, memory_dir or batch_dir)
    record = {
        "source_config": source,
        "reference_config": str(reference),
        "enabled_guards": enabled,
        "guard_modes": {name: raw[name].get("mode") for name in enabled},
        "agrail_memory": memory,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record_manifest(batch_dir, label, record)
    return record


def finalize(batch_dir: pathlib.Path) -> None:
    """Record the final hash of every campaign AGrail memory file."""
    path = batch_dir / MANIFEST
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in manifest.values():
        memory = (record or {}).get("agrail_memory") if isinstance(record, dict) else None
        if memory and pathlib.Path(memory["path"]).exists():
            memory["sha256_final"] = _sha256(pathlib.Path(memory["path"]))
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("configure", help="write a campaign config")
    prepare.add_argument("source", type=pathlib.Path)
    prepare.add_argument("output", type=pathlib.Path)
    prepare.add_argument("batch_dir", type=pathlib.Path)
    prepare.add_argument("--label", default="campaign")
    prepare.add_argument(
        "--memory-dir", type=pathlib.Path,
        help="directory for the fresh AGrail memory (default: batch_dir)",
    )
    prepare.add_argument(
        "--keep-guards", action="store_true",
        help="keep the source guard set (the source is the reference itself)",
    )
    done = commands.add_parser("finalize", help="record final memory hashes")
    done.add_argument("batch_dir", type=pathlib.Path)
    memory = commands.add_parser(
        "memory", help="create a campaign AGrail memory (kept on resume); print its path"
    )
    memory.add_argument("batch_dir", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "finalize":
        finalize(args.batch_dir)
        return
    if args.command == "memory":
        print(campaign_memory(args.batch_dir))
        return
    raw = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    reference = args.source if args.keep_guards else REFERENCE
    configure(raw, args.batch_dir, args.label, args.memory_dir, str(args.source), reference)
    args.output.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
