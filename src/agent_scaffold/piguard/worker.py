"""PIGuard classifier, run by the isolated interpreter (/opt/piguard-venv).

This file imports only upstream's inference stack, never ``agent_scaffold``.
Loading and inference follow the authors' ``inference_examples.py`` in the
model repository: ``AutoModelForSequenceClassification`` with
``trust_remote_code=True`` and a ``text-classification`` pipeline with
``truncation=True, max_length=512``; the verdict is the pipeline's top label
(``benign`` / ``injection``). Scores of both labels and the untruncated token
count are reported as well, for analysis only.

Protocol: one JSON object per line. The first output line is
``{"ready": true, ...}``; each input ``{"texts": [...]}`` is answered with
``{"results": [...]}`` (or ``{"error": ...}``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=args.max_length,
        device=torch.device("cpu"),
    )
    revision_file = Path(args.model).parent / "REVISION"
    print(json.dumps({
        "ready": True,
        "revision": revision_file.read_text().strip() if revision_file.is_file() else "",
        "labels": model.config.id2label,
        "max_length": args.max_length,
    }), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            texts = [str(text) for text in json.loads(line)["texts"]]
            # top_k=None returns every label's score; the highest one is the
            # pipeline's default (top_k=1) verdict.
            scores = classifier(texts, top_k=None)
            results = []
            for text, every in zip(texts, scores):
                best = max(every, key=lambda item: item["score"])
                tokens = len(tokenizer(text, truncation=False)["input_ids"])
                results.append({
                    "label": best["label"],
                    "score": float(best["score"]),
                    "scores": {item["label"]: float(item["score"]) for item in every},
                    "tokens": tokens,
                    "truncated": tokens > args.max_length,
                })
            print(json.dumps({"results": results}), flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the guard as an error
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
