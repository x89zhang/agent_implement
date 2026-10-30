"""Serve the local PIGuard classifier using the mini_swe environment."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from threading import Lock
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn


class ClassifyRequest(BaseModel):
    texts: list[str]


def create_app(model_path: str, max_length: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check the mini_swe PyTorch installation "
                "and GPU_ID/CUDA_VISIBLE_DEVICES."
            )
        device = torch.device("cuda:0")
        logging.getLogger("uvicorn.error").info(
            "Loading PIGuard from %s on %s", model_path,
            torch.cuda.get_device_name(device),
        )
        app.state.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True,
        )
        app.state.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        ).to(device).eval()
        app.state.device = device
        app.state.lock = Lock()
        yield

    app = FastAPI(title="PIGuard", lifespan=lifespan)
    model_info = {
        "id": "PIGuard",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }

    @app.get("/v1/models")
    def list_models():
        # Model discovery for clients that probe OpenAI-style server URLs.
        # Inference is still served through /classify, not chat completions.
        return {"object": "list", "data": [model_info]}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path, "device": "cuda:0"}

    @app.post("/classify")
    def classify(request: ClassifyRequest):
        results = []
        # Process one text at a time to bound GPU memory for large requests.
        with app.state.lock, torch.inference_mode():
            for text in request.texts:
                inputs = app.state.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=max_length,
                ).to(app.state.device)
                probabilities = app.state.model(**inputs).logits.softmax(-1)[0]
                scores = probabilities.cpu().tolist()
                results.append(sorted(
                    [
                        {"label": app.state.model.config.id2label[index],
                         "score": score}
                        for index, score in enumerate(scores)
                    ],
                    key=lambda item: item["score"], reverse=True,
                ))
        return {"results": results}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/data/xiaoliang/models/PIGuard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if not Path(args.model, "config.json").is_file():
        parser.error(f"Model config not found: {args.model}/config.json")
    if not 1 <= args.max_length <= 512:
        parser.error("--max-length must be between 1 and 512")
    uvicorn.run(create_app(args.model, args.max_length), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
