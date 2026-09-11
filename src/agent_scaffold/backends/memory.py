"""Only declared memory surfaces may cross a phase boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

SURFACES = ("MEMORY.md", "USER.md")


def copy_memory(source: Path | None, target_home: Path):
    target = target_home / "memories"
    target.mkdir(parents=True, exist_ok=True)
    if source is None:
        return
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"Memory source must be a real directory: {source}")
    for name in SURFACES:
        path = source / name
        if path.is_symlink():
            raise ValueError(f"Memory symlinks are not supported: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Memory surface is not a file: {path}")
            (target / name).write_bytes(path.read_bytes())


def manifest(home: Path):
    result = {}
    for name in SURFACES:
        path = home / "memories" / name
        if path.is_symlink():
            raise ValueError(f"Unexpected memory symlink: {path}")
        data = path.read_bytes() if path.exists() else None
        result[name] = {
            "exists": data is not None,
            "bytes": len(data or b""),
            "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
        }
    return result
