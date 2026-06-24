from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from moondream_mini import MiniConfig, MiniMoondream
from moondream_mini.prompts import build_prompt, extract_answer, resolve_label_space


def parse_args():
    p = argparse.ArgumentParser(description="Infer with Moondream-mini")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task-type", type=str, default=None, choices=["yes_no", "count", "location"])
    p.add_argument("--label-space", type=str, default=None, choices=["yes_no", "count_4", "location_3"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg_dict["vocab_size"] = tokenizer.get_vocab_size()
    if "image_size" not in cfg_dict:
        cfg_dict["image_size"] = 224
    if "num_image_tokens" not in cfg_dict:
        cfg_dict["num_image_tokens"] = (cfg_dict["image_size"] // 16) ** 2 + 1
    model = MiniMoondream(MiniConfig(**cfg_dict)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    image_size = int(cfg_dict.get("image_size", 224))
    img = Image.open(args.image).convert("RGB").resize((image_size, image_size))
    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    task_type, label_space = resolve_label_space(args.task_type, args.label_space, args.question)
    prompt = build_prompt(args.question, label_space)
    generated = model.generate(
        arr,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(f"task_type: {task_type}")
    print(f"label_space: {label_space}")
    print(f"answer: {extract_answer(generated, label_space)}")
    print(f"raw_output: {generated}")


if __name__ == "__main__":
    main()
