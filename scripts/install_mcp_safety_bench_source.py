"""Install a pinned MCP-SafetyBench source tree without executing repository code."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: install_mcp_safety_bench_source.py REVISION TARGET")
    revision, target_text = sys.argv[1:]
    target = Path(target_text)
    url = f"https://github.com/xjzzzzzzzz/MCPSafety/archive/{revision}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="mcp-safety-bench-source-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
        unpacked = Path(temporary) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            root = unpacked.resolve()
            for member in stream.getmembers():
                destination = (unpacked / member.name).resolve()
                if root not in destination.parents and destination != root:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            stream.extractall(unpacked, filter="data")
        roots = [path for path in unpacked.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("The archive must contain exactly one source root")
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(roots[0]), target)
    (target / ".mcp-safety-bench-revision").write_text(revision + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
