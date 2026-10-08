"""Threshold calibration on held-out safe runs (ProbGuard paper §5.2).

The paper calibrates θ by the false-positive rate "on safe execution traces
where no actual violations occur". Here those are clean runs captured for the
task but left out of DTMC learning. A run alarms when any executed step's
P(F unsafe) reaches θ (``middleware.py`` ``>=``), so each run is summarized by
its highest step probability, computed exactly as at runtime: skill steps are
not abstracted and states unseen in training are skipped.
"""

from __future__ import annotations

import math
from typing import Any

from .abstraction import PredicateAbstraction
from .model import JsonDTMC


def run_max_probability(
    abstraction: PredicateAbstraction,
    dtmc: JsonDTMC,
    unsafe_ids: set[int],
    steps: list[dict[str, Any]],
    bound: int = -1,
) -> float | None:
    """Highest P(F unsafe) over a run's steps; ``None`` when every state is unseen."""
    best = None
    for index in range(len(steps)):
        encoded = abstraction.encode(steps[: index + 1])
        if encoded not in dtmc.state_index:
            continue
        probability = dtmc.probability_to_unsafe(dtmc.state_index[encoded], unsafe_ids, bound)
        best = probability if best is None else max(best, probability)
    return best


def calibrate_threshold(maxima: list[float | None], target_fpr: float) -> dict[str, Any]:
    """Smallest θ (``>=`` alarms) with at most ``target_fpr`` of runs alarming.

    A run without any seen state never alarms and counts as 0.0. With n runs,
    floor(target_fpr * n) may alarm; θ is placed just above the next highest
    run maximum. θ above 1.0 means no threshold meets the target.
    """
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    values = sorted((value or 0.0 for value in maxima), reverse=True)
    if not values:
        raise ValueError("calibration needs at least one held-out clean run")
    allowed = math.floor(target_fpr * len(values) + 1e-9)
    threshold = math.nextafter(values[allowed], math.inf)
    return {
        "threshold": threshold,
        "achievable": threshold <= 1.0,
        "target_fpr": target_fpr,
        "runs": len(values),
        "alarms_allowed": allowed,
        "run_max_probabilities": values,
    }
