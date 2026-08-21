from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


REVISION = "1f561dccf92d55302368fa67679b4ba9d9c8fdc4"
FILES = (
    "agent_task.jsonl",
    "all_normal_tools.jsonl",
    "all_attack_tools.jsonl",
)


def main() -> None:
    target = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/opt/agent-security-bench/data"
    )
    target.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/agiresearch/ASB/{REVISION}/data"
    for name in FILES:
        destination = target / name
        with urllib.request.urlopen(f"{base}/{name}", timeout=60) as response:
            destination.write_bytes(response.read())


if __name__ == "__main__":
    main()
