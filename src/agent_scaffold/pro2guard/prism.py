from __future__ import annotations

import re
import subprocess
from pathlib import Path


def query_prism_probability(
    *,
    prism_bin: str,
    dtmc_path: str,
    current_state: int | str,
    unsafe_states: list[str],
    timeout_seconds: int,
) -> float:
    unsafe_expr = _unsafe_expression(unsafe_states)
    formula = f'P=? [ F ({unsafe_expr}) ]'
    cmd = [prism_bin, str(Path(dtmc_path)), "-pf", formula, "-const", f"init={current_state}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    match = re.search(r"Result:\s*([0-9.eE+-]+)", result.stdout)
    if not match:
        detail = (result.stderr or result.stdout or "no PRISM output").strip()
        raise RuntimeError(f"Could not parse PRISM probability: {detail[:500]}")
    return float(match.group(1))


def _unsafe_expression(unsafe_states: list[str]) -> str:
    if not unsafe_states:
        return "false"
    parts = []
    for state in unsafe_states:
        text = str(state).strip()
        if re.fullmatch(r"\d+", text):
            parts.append(f"s={text}")
        else:
            parts.append(text)
    return " | ".join(parts)
