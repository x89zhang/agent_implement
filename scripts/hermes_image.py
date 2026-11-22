#!/usr/bin/env python3
"""Print the Hermes image tag a config runs in, without building it.

Supplement replays need the image of an all-monitors batch: by default they
take the newest image, which after a monitor-off screen lacks the monitors'
dependencies. Exits non-zero if the image has not been built.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agent_scaffold.backends import container  # noqa: E402
from agent_scaffold.config import load_config  # noqa: E402


class _Tagged(Exception):
    pass


def image_for(config: pathlib.Path) -> str:
    def inspect(command, workspace):
        raise _Tagged(command[-1])

    # ensure_hermes_image computes the tag, then inspects it; stop there.
    with mock.patch.object(container, "_run_checked", inspect):
        try:
            container.ensure_hermes_image(load_config(config), REPO)
        except _Tagged as tagged:
            return str(tagged)
    raise RuntimeError(f"could not determine the image for {config}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    image = image_for(parser.parse_args().config)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        print(f"image not built: {image}", file=sys.stderr)
        return 1
    print(image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
