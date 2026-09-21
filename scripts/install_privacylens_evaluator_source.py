"""Install a pinned SALT-NLP PrivacyLens evaluator source tree."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: install_privacylens_evaluator_source.py REVISION TARGET"
        )
    revision, target_text = sys.argv[1:]
    target = Path(target_text)
    url = f"https://github.com/SALT-NLP/PrivacyLens/archive/{revision}.tar.gz"
    with tempfile.TemporaryDirectory(
        prefix="privacylens-evaluator-source-"
    ) as temporary:
        archive = Path(temporary) / "privacylens.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            stream.extractall(extracted, filter="data")
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(
                "PrivacyLens archive did not contain exactly one source root"
            )
        required = roots[0] / "evaluation" / "evaluate_final_action.py"
        if not required.is_file():
            raise RuntimeError(
                f"PrivacyLens evaluator source is incomplete: {required}"
            )
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(roots[0]), target)
    (target / ".privacylens-evaluator-revision").write_text(
        revision + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
