from __future__ import annotations

import base64
import io
import mimetypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from moondream_mini import MiniConfig, MiniMoondream
from moondream_mini.prompts import build_prompt, extract_answer, normalize_text, resolve_label_space

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="Moondream Mini PPE API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MOONDREAM_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferRequest(BaseModel):
    image: str = Field(..., description="Data URL or base64 image string")
    question: str
    task_type: str | None = None
    label_space: str | None = None
    max_new_tokens: int = 16
    temperature: float = 0.0
    repetition_penalty: float = 1.08
    no_repeat_ngram_size: int = 3


class InferResponse(BaseModel):
    raw_output: str
    final_answer: str
    label_space: str
    prompt: str
    task_type: str


class GalleryItem(BaseModel):
    id: str
    name: str
    path: str
    url: str
    mime: str


class GalleryResponse(BaseModel):
    items: list[GalleryItem]


def required_path(env_name: str, default: str | None = None) -> Path:
    value = os.environ.get(env_name, default)
    if not value:
        raise RuntimeError(f"{env_name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    tokenizer_dir = required_path("MOONDREAM_TOKENIZER", "artifacts/tokenizer")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


@lru_cache(maxsize=1)
def get_model() -> tuple[MiniMoondream, torch.device, dict[str, Any]]:
    checkpoint = required_path("MOONDREAM_CHECKPOINT")
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    device = torch.device(os.environ.get("MOONDREAM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    tokenizer = get_tokenizer()
    cfg_dict["vocab_size"] = tokenizer.get_vocab_size()
    cfg_dict.setdefault("image_size", 224)
    cfg_dict.setdefault("num_image_tokens", (cfg_dict["image_size"] // 16) ** 2 + 1)
    model = MiniMoondream(MiniConfig(**cfg_dict)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, device, cfg_dict


def decode_image_payload(data: str) -> Image.Image:
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def project_data_roots() -> list[Path]:
    roots: list[Path] = []
    data_root = Path(os.environ.get("MOONDREAM_DATA_ROOT", PROJECT_ROOT / "data"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if data_root.exists():
        roots.append(data_root)
    return roots


def find_images() -> list[Path]:
    seen: set[str] = set()
    items: list[Path] = []
    for root in project_data_roots():
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
                key = str(candidate.resolve())
                if key in seen:
                    continue
                seen.add(key)
                items.append(candidate)
    return sorted(items)


@app.get("/api/gallery", response_model=GalleryResponse)
def gallery() -> GalleryResponse:
    items: list[GalleryItem] = []
    for idx, path in enumerate(find_images()[:256]):
        rel = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        items.append(GalleryItem(id=str(idx), name=path.name, path=rel, url=f"/api/image?path={quote(rel)}", mime=mime))
    return GalleryResponse(items=items)


@app.get("/api/image")
def image(path: str):
    img_path = (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if PROJECT_ROOT not in img_path.parents and img_path != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)


@app.post("/api/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    try:
        image = decode_image_payload(payload.image)
        model, device, cfg = get_model()
        tokenizer = get_tokenizer()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_type, label_space = resolve_label_space(payload.task_type, payload.label_space, payload.question)
    image_size = int(cfg.get("image_size", 224))
    image = image.resize((image_size, image_size))
    arr = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    question = normalize_text(payload.question)
    prompt = build_prompt(question, label_space)
    with torch.no_grad():
        generated = model.generate(
            arr,
            tokenizer,
            prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            repetition_penalty=payload.repetition_penalty,
            no_repeat_ngram_size=payload.no_repeat_ngram_size,
        )
    return InferResponse(
        raw_output=generated,
        final_answer=extract_answer(generated, label_space),
        label_space=label_space,
        prompt=prompt,
        task_type=task_type,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
