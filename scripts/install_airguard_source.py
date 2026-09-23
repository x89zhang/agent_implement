"""Install only AIRGuard's source package from an immutable revision."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: install_airguard_source.py REVISION TARGET")
    revision, target_text = sys.argv[1:]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("AIRGuard revision must be a full lowercase commit SHA")
    target = Path(target_text)
    if target.exists():
        raise ValueError(f"AIRGuard target already exists: {target}")
    url = f"https://github.com/Sophie508/AIRGuard/archive/{revision}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="airguard-source-") as temporary:
        archive = Path(temporary) / "airguard.tar.gz"
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
                    raise RuntimeError(f"unsafe AIRGuard archive member: {member.name}")
                parts = Path(member.name).parts
                if len(parts) >= 3 and parts[1:3] == ("src", "airguard"):
                    members.append(member)
            stream.extractall(unpacked, members=members, filter="data")
        roots = list(unpacked.iterdir())
        if len(roots) != 1 or not (roots[0] / "src" / "airguard" / "guard.py").is_file():
            raise RuntimeError("AIRGuard archive has no guard source tree")
        target.mkdir(parents=True)
        shutil.move(str(roots[0] / "src" / "airguard"), target / "airguard")
    (target / ".airguard-revision").write_text(revision + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
