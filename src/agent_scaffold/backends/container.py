"""Build an isolated Hermes interpreter and reuse the project's Docker lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml

from ..config import load_config_mapping
from ..container_runtime import (
    _clawsentry_managed, _ensure_image, _image_has_clawsentry,
    _run_checked, run_once_in_container,
)
from ..skills import load_enabled_skills


def ensure_hermes_image(cfg, workspace):
    settings = cfg.execution.hermes
    repo = Path(settings.repo_path)
    if (repo / ".env").exists():
        raise ValueError("Use a Hermes checkout without a repository .env")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError(
            "Automatic Hermes image builds require a clean checkout; commit changes first"
        )
    if settings.expected_commit and commit != settings.expected_commit:
        raise ValueError(
            f"Hermes revision mismatch: expected {settings.expected_commit}, found {commit}"
        )
    base = copy.deepcopy(cfg)
    args = base.container.build_args
    for section, argument in (
        ("agentdojo", "INSTALL_AGENTDOJO"),
        ("agent_security_bench", "INSTALL_AGENT_SECURITY_BENCH"),
        ("agentharm", "INSTALL_AGENTHARM"),
        ("privacylens_live", "INSTALL_PRIVACYLENS_LIVE"),
        ("llamafirewall", "INSTALL_LLAMA_FIREWALL"),
        ("agentspec", "INSTALL_AGENTSPEC"),
        ("progent", "INSTALL_PROGENT"),
        ("adr", "INSTALL_ADR"),
    ):
        if getattr(cfg, section).enabled:
            args[argument] = "true"
    if _clawsentry_managed(cfg):
        args["INSTALL_CLAWSENTRY"] = "true"
    dockerfile = Path(cfg.container.dockerfile)
    if not dockerfile.is_absolute():
        dockerfile = workspace / dockerfile
    digest = hashlib.sha256()
    for file in [
        dockerfile,
        workspace / "integrations/hermes/Dockerfile",
        workspace / "scripts/install_asb_source.py",
        workspace / "scripts/install_adr_source.py",
        workspace / "scripts/install_privacylens_live_source.py",
        workspace / "scripts/install_privacylens_evaluator_source.py",
        *sorted(workspace.glob("requirements*.txt")),
    ]:
        digest.update(file.name.encode() + file.read_bytes())
    digest.update(json.dumps(args, sort_keys=True).encode())
    base.container.image = cfg.container.image + "-base-" + digest.hexdigest()[:12]
    digest.update(commit.encode())
    image = cfg.container.image + "-hermes-" + digest.hexdigest()[:12]
    if _run_checked(["docker", "image", "inspect", image], workspace).returncode == 0:
        if not _clawsentry_managed(cfg) or _image_has_clawsentry(image, workspace):
            return image
    if not cfg.container.auto_build:
        raise RuntimeError(
            f"Hermes image {image} missing and container.auto_build is false"
        )
    _ensure_image(base, workspace)
    with tempfile.TemporaryDirectory(prefix="hermes-image-") as temporary:
        context = Path(temporary)
        clone = _run_checked(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--no-local",
                str(repo),
                str(context / "hermes-source"),
            ],
            workspace,
        )
        if clone.returncode:
            raise RuntimeError(f"Cannot stage Hermes source: {clone.stderr}")
        staged_commit = subprocess.check_output(
            ["git", "-C", str(context / "hermes-source"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if staged_commit != commit:
            raise RuntimeError("Hermes checkout changed while staging the Docker build")
        for source, target in (
            (workspace / "integrations/hermes/Dockerfile", "Dockerfile"),
            (
                workspace / "requirements-hermes-bridge.txt",
                "requirements-hermes-bridge.txt",
            ),
        ):
            (context / target).write_bytes(source.read_bytes())
        result = _run_checked(
            [
                "docker",
                "build",
                "-t",
                image,
                "--build-arg",
                f"BASE_IMAGE={base.container.image}",
                str(context),
            ],
            workspace,
        )
        if result.returncode:
            raise RuntimeError(
                f"Hermes image build failed:\n{result.stdout}\n{result.stderr}"
            )
    return image


def prepare_container_config(cfg, cfg_path, workspace, run_dir):
    """Resolve host paths before moving YAML; external inputs get read-only mounts."""
    raw = load_config_mapping(cfg_path)
    raw.pop("harness", None)
    mounts = {}
    workdir = cfg.container.workdir.rstrip("/") or "/workspace"
    config_dir = Path(cfg.config_dir)

    def remap(value, key=""):
        if isinstance(value, dict):
            return {k: remap(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v, key) for v in value]
        if not isinstance(value, str) or not value:
            return value
        candidate = Path(value).expanduser()
        is_path = key.endswith(("_path", "_dir", "_file")) or key in {
            "path",
            "policy",
            "plugin_config",
            "model_path",
            "dtmc_path",
            "detection_root",
        }
        if not candidate.is_absolute():
            if not is_path:
                return value
            candidate = config_dir / candidate
            if not candidate.exists():
                candidate = workspace / value
        if not candidate.exists():
            return value
        candidate = candidate.resolve()
        try:
            return str(Path(workdir) / candidate.relative_to(workspace))
        except ValueError:
            host = str(candidate)
            if host not in mounts:
                mounts[host] = f"/opt/project-inputs/{len(mounts)}/{candidate.name}"
            return mounts[host]

    # Model/interpreter are supplied by the image, never mounted from host venvs.
    raw.setdefault("execution", {}).setdefault("hermes", {})
    if isinstance(raw.get("adr"), dict) and raw["adr"].get("enabled"):
        # These paths belong to the image, not to the host-path remapper.
        raw["adr"]["detection_root"] = ""
        raw["adr"]["python_executable"] = ""
    raw["execution"]["hermes"].update(
        repo_path="/opt/hermes-agent", python_executable="/opt/hermes-venv/bin/python"
    )
    raw.setdefault("skills", {})["enabled"] = [
        {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
            "requires_tools": skill.requires_tools,
            "priority": skill.priority,
        }
        for skill in load_enabled_skills(cfg)
        if skill.path != "agentdojo://skill-injection"
    ]
    raw = remap(raw)
    if isinstance(raw.get("adr"), dict) and raw["adr"].get("enabled"):
        raw["adr"]["detection_root"] = "/opt/adr/Detection"
        raw["adr"]["python_executable"] = "/opt/adr-venv/bin/python"
    raw["execution"]["hermes"].update(
        repo_path="/opt/hermes-agent", python_executable="/opt/hermes-venv/bin/python"
    )
    path = run_dir / "hermes.container.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    path.chmod(0o600)
    return path, [(host, target) for host, target in mounts.items()]


def run_hermes_in_container(
    cfg,
    cfg_path,
    user_input,
    context_messages,
    resume_messages,
    workspace_root,
    run_dir,
):
    cfg = copy.deepcopy(cfg)

    def collect_env(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_env") and isinstance(item, str) and item:
                    if item not in cfg.container.env:
                        cfg.container.env.append(item)
                else:
                    collect_env(item)
        elif isinstance(value, list):
            for item in value:
                collect_env(item)

    collect_env(asdict(cfg))
    cfg.container.image = ensure_hermes_image(cfg, workspace_root)
    config, mounts = prepare_container_config(cfg, cfg_path, workspace_root, run_dir)
    # The child selects /opt/hermes-venv; reuse logging, observer gates and cleanup.
    return run_once_in_container(
        cfg,
        str(config),
        user_input,
        context_messages,
        resume_messages,
        workspace_root,
        run_dir,
        extra_mounts=mounts,
        image_ready=True,
        run_as_host_user=True,
    )
