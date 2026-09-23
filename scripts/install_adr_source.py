"""Install a pinned copy of ADR Detection without running its package setup."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: install_adr_source.py REVISION TARGET")
    revision, target_text = sys.argv[1:]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("ADR revision must be a full lowercase commit SHA")
    target = Path(target_text)
    if target.exists():
        raise ValueError(f"ADR target already exists: {target}")
    url = f"https://github.com/uber/ADR/archive/{revision}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="adr-source-") as temporary:
        archive = Path(temporary) / "adr.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
        unpacked = Path(temporary) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            root = unpacked.resolve()
            members = []
            for member in stream.getmembers():
                destination = (unpacked / member.name).resolve()
                if root not in destination.parents and destination != root:
                    raise RuntimeError(f"unsafe ADR archive member: {member.name}")
                parts = Path(member.name).parts
                if len(parts) >= 2 and parts[1] == "Detection":
                    members.append(member)
            stream.extractall(unpacked, members=members, filter="data")
        roots = list(unpacked.iterdir())
        if len(roots) != 1 or not (roots[0] / "Detection").is_dir():
            raise RuntimeError("ADR archive has no Detection source tree")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0] / "Detection"), target)
    (target / ".adr-revision").write_text(revision + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
