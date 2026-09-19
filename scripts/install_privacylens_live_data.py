from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REVISION = "994ac15db6fff8a5131bbf5a26e84e352e676796"
SOURCE = "misc/PrivacyInAction/PrivacyLens-Live/MCP-2Tools/baseline/filtered_data.json"


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/privacylens-live/data")
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/microsoft/ACV/{REVISION}/{SOURCE}"
    with urllib.request.urlopen(url, timeout=60) as response:
        (target / "filtered_data.json").write_bytes(response.read())


if __name__ == "__main__":
    main()
