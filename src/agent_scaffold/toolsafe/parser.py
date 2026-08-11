from __future__ import annotations

import json
import re
from typing import Any


_SCORE_KEYS = ("risk_score", "risk rating", "risk_rating", "score", "judgment")
_SCORE_PATTERN = r"(?:0(?:\.0)?|0\.5|1(?:\.0)?)"


def parse_guard_response(text: str) -> tuple[float, str]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("ToolSafe returned an empty response")

    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        for key in _SCORE_KEYS:
            if key in parsed:
                score = _validate_score(parsed[key])
                reason = str(
                    parsed.get("reason")
                    or parsed.get("analysis")
                    or parsed.get("think")
                    or raw
                ).strip()
                return score, reason

    patterns = (
        rf"<Judgment>\s*({_SCORE_PATTERN})\s*<(?:/|\\)Judgment>",
        rf"(?:risk[ _-]?(?:rating|score)|judgment)\s*[:=]\s*({_SCORE_PATTERN})",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            score = _validate_score(match.group(1))
            think = re.search(
                r"<Think>\s*(.*?)\s*<(?:/|\\)Think>",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return score, (think.group(1).strip() if think else raw)

    if re.fullmatch(_SCORE_PATTERN, raw):
        return _validate_score(raw), raw
    raise ValueError("ToolSafe response does not contain a valid risk score")


def _parse_json(text: str) -> Any:
    candidate = text
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        return json.loads(candidate)
    except (TypeError, ValueError):
        return None


def _validate_score(value: Any) -> float:
    score = float(value)
    if score not in {0.0, 0.5, 1.0}:
        raise ValueError("ToolSafe risk score must be 0.0, 0.5, or 1.0")
    return score
