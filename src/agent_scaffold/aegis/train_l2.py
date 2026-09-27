"""Train the AEGIS L2 XGBoost model with the paper's recipe and data.

This reuses the upstream research code unchanged (``paper/Aegis/research``):
``benchmark.schema`` records, ``cascade.pipeline.CascadePipeline`` with
``l2_mode="xgboost"`` and ``fit_l2_supervised`` (``L2XGBoost``: 300 trees,
depth 6, lr 0.1, auto ``scale_pos_weight``, thresholds calibrated on a 15%
validation split for 1% FPR / 5% FNR). The stratified 50/50 fit/test split is
``cascade/run_cascade.py``'s xgboost branch with ``--seed 0``.

The training data is upstream's aegis-bench, built with upstream's
``REPRODUCE.md`` step 1 (InjecAgent, ToolEmu, OWASP payloads, AEGIS self-suite;
5525 records). It contains none of this project's evaluation benchmarks::

    cd <copy of paper/Aegis>/research
    python -m benchmark.scripts.download_all --only injecagent toolemu owasp aegis_self
    python -m benchmark.scripts.build_owasp_payloads
    python -m benchmark.build

Then, with ``xgboost`` installed (see requirements-aegis.txt)::

    PYTHONPATH=src python -m agent_scaffold.aegis.train_l2 \
        --research-root <copy>/research --out models/aegis/l2_xgboost.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_head(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("models/aegis/l2_xgboost.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2-fit-frac", type=float, default=0.5)
    parser.add_argument("--xgb-target-fpr", type=float, default=0.01)
    parser.add_argument("--xgb-target-fnr", type=float, default=0.05)
    args = parser.parse_args(argv)

    research = args.research_root.resolve()
    bench = (args.bench or research / "benchmark" / "data" / "aegis-bench.jsonl").resolve()
    sys.path.insert(0, str(research))
    import numpy as np
    import xgboost
    from benchmark.schema import BenchRecord, Label  # type: ignore
    from cascade.pipeline import CascadePipeline, CascadeThresholds  # type: ignore

    records = [BenchRecord.model_validate_json(line) for line in bench.open()]
    # run_cascade.py, --l2-mode xgboost branch.
    rng = random.Random(args.seed)
    benign = [r for r in records if r.label == Label.BENIGN]
    malicious = [r for r in records if r.label == Label.MALICIOUS]
    rng.shuffle(benign)
    rng.shuffle(malicious)
    fit_set = (benign[: int(len(benign) * args.l2_fit_frac)]
               + malicious[: int(len(malicious) * args.l2_fit_frac)])
    fit_ids = {r.id for r in fit_set}
    test_set = [r for r in records if r.id not in fit_ids]

    cascade = CascadePipeline(
        l1=None, l3=None, thresholds=CascadeThresholds(),
        use_l1=False, use_l2=True, use_l3=False,
        l2_mode="xgboost", random_state=args.seed,
    )
    stats = cascade.fit_l2_supervised(
        fit_set, target_fpr=args.xgb_target_fpr, target_fnr=args.xgb_target_fnr,
    )
    tau_high, tau_low = cascade.thresholds.tau_high, cascade.thresholds.tau_low

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cascade._xgb._model.save_model(str(args.out))

    # Held-out L2-only resolution, for comparison with the paper's Table.
    scores = np.array([cascade._score_l2(r) for r in test_set])
    labels = np.array([r.label == Label.MALICIOUS for r in test_set])
    blocked, allowed = scores >= tau_high, scores < tau_low
    escalated = ~(blocked | allowed)

    raw = research / "benchmark" / "data" / "raw"
    metadata = {
        "tau_high": tau_high,
        "tau_low": tau_low,
        "feature_names": list(stats["feature_importances"]),
        "training": {
            "recipe": "research/cascade/run_cascade.py --l2-mode xgboost (seed split) + "
                      "CascadePipeline.fit_l2_supervised",
            "seed": args.seed,
            "l2_fit_frac": args.l2_fit_frac,
            "target_fpr": args.xgb_target_fpr,
            "target_fnr": args.xgb_target_fnr,
            "stats": stats,
            "xgboost_version": xgboost.__version__,
            "numpy_version": np.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        "data": {
            "bench_sha256": hashlib.sha256(bench.read_bytes()).hexdigest(),
            "n_records": len(records),
            "n_fit": len(fit_set),
            "n_test": len(test_set),
            "sources": sorted({r.source for r in records}),
            "aegis_commit": _git_head(research),
            "injecagent_commit": _git_head(raw / "_clone_injecagent"),
            "toolemu_commit": _git_head(raw / "_clone_toolemu"),
        },
        "heldout_l2": {
            "block_rate_malicious": float(blocked[labels].mean()),
            "block_rate_benign": float(blocked[~labels].mean()),
            "escalated_fraction": float(escalated.mean()),
            "missed_malicious_allowed": int((allowed & labels).sum()),
        },
    }
    meta_path = args.out.with_name(args.out.stem + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # The runtime may evaluate the saved trees without xgboost; check parity.
    from agent_scaffold.aegis.cascade import L2Model

    model = L2Model(args.out, metadata)
    model._booster = None
    model._load_trees(json.loads(args.out.read_text(encoding="utf-8")))
    parity = max(abs(model.score(r.tool_call.arguments) - s)
                 for r, s in zip(test_set, scores))
    if parity > 0.0:
        raise SystemExit(f"saved-tree evaluator differs from xgboost by {parity}")
    print(json.dumps({"tau_high": tau_high, "tau_low": tau_low,
                      "parity": parity, **metadata["heldout_l2"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
