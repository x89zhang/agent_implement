from __future__ import annotations

import os
import time
from pathlib import Path


def wait_for_start_gate() -> None:
    start_file = os.environ.get("AGENTSIGHT_START_FILE")
    if not start_file:
        return

    # AgentSight resolves docker:// targets from /proc/<pid>/maps. Importing
    # ssl before the gate guarantees dynamically-linked Python containers have
    # loaded libssl before the host-side observer tries to resolve its uprobe.
    import ssl

    _ = ssl.OPENSSL_VERSION
    ready_file = os.environ.get("AGENTSIGHT_READY_FILE")
    if ready_file:
        ready_path = Path(ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.touch()
    deadline = time.monotonic() + float(os.environ.get("AGENTSIGHT_START_TIMEOUT", "60"))
    while not Path(start_file).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for AgentSight start gate: {start_file}")
        time.sleep(0.05)
