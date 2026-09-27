"""PRISM backend, following upstream ``monitor_dtmc.py`` (build_pctl / check_reachability)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


def build_pctl(unsafe_indices: list[int], bound: int = -1) -> str:
    if len(unsafe_indices) == 1:
        state_expr = f"s={unsafe_indices[0]}"
    else:
        state_expr = "(" + "|".join(f"s={u}" for u in unsafe_indices) + ")"
    if bound > 0:
        return f"P=? [ F<={bound} ({state_expr}) ]"
    return f"P=? [ F ({state_expr}) ]"


def query_prism_probability(
    *,
    prism_bin: str,
    dtmc_path: str,
    current_state: int,
    unsafe_indices: list[int],
    bound: int = -1,
    timeout_seconds: int = 10,
) -> float:
    if not unsafe_indices:
        return 0.0
    model_txt = Path(dtmc_path).read_text(encoding="utf-8")
    # Upstream rewrites the model's init state to the current abstract state.
    updated = re.sub(r"init\s+\d+", f"init {int(current_state)}", model_txt)
    with tempfile.NamedTemporaryFile("w", suffix=".prism", delete=False) as stream:
        stream.write(updated)
        tmp_path = stream.name
    try:
        result = subprocess.run(
            [prism_bin, tmp_path, "-pf", build_pctl(unsafe_indices, bound)],
            capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    match = re.search(r"Result:\s*([0-9.eE+-]+)", result.stdout)
    if not match:
        detail = (result.stderr or result.stdout or "no PRISM output").strip()
        raise RuntimeError(f"Could not parse PRISM probability: {detail[:500]}")
    return float(match.group(1))
