"""Install a pinned PrivacyLens-Live source subtree without executing it."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: install_privacylens_live_source.py REVISION TARGET")
    revision, target_text = sys.argv[1:]
    target = Path(target_text)
    url = f"https://github.com/microsoft/ACV/archive/{revision}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="privacylens-live-source-") as temporary:
        archive = Path(temporary) / "acv.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
        staged = Path(temporary) / "staged"
        staged.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            members = []
            for member in stream.getmembers():
                parts = Path(member.name).parts
                if (
                    len(parts) < 5
                    or "/".join(parts[1:4]) != "misc/PrivacyInAction/PrivacyLens-Live"
                ):
                    continue
                relative = Path(*parts[4:])
                if not relative.parts:
                    continue
                destination = (staged / relative).resolve()
                if staged.resolve() not in destination.parents:
                    raise RuntimeError(
                        f"unsafe PrivacyLens-Live archive member: {member.name}"
                    )
                member.name = str(relative)
                members.append(member)
            if not members:
                raise RuntimeError(
                    f"PrivacyLens-Live subtree not found in ACV revision {revision}"
                )
            stream.extractall(staged, members=members, filter="data")
        required = staged / "MCP-2Tools" / "baseline" / "filtered_data.json"
        if not required.is_file():
            raise RuntimeError(f"PrivacyLens-Live source is incomplete: {required}")
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(staged), target)
    (target / ".privacylens-live-revision").write_text(
        revision + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
