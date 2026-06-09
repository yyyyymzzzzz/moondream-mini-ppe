from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from moondream_mini import MiniConfig, MiniMoondream
from moondream_mini.prompts import build_prompt, extract_answer, normalize_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Moondream-mini on PPE VQA JSONL data")
    p.add_argument("--test-jsonl", type=Path, default=Path("data/moondream_ppe_vqa/test.jsonl"))
    p.add_argument("--image-root", type=Path, default=Path("data/moondream_ppe_vqa"))
    p.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="Optional max number of examples to run; 0 means all")
    p.add_argument("--output-jsonl", type=Path, default=None, help="Optional path for per-example predictions")
    return p.parse_args()


def normalize_label(s: str) -> str:
    return normalize_text(s).lower()


def load_model(checkpoint: Path, tokenizer: Tokenizer, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg_dict["vocab_size"] = tokenizer.get_vocab_size()
    cfg_dict.setdefault("image_size", 224)
    cfg_dict.setdefault("num_image_tokens", (cfg_dict["image_size"] // 16) ** 2 + 1)
    model = MiniMoondream(MiniConfig(**cfg_dict)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, int(cfg_dict["image_size"])


def resolve_image_path(image: str, image_root: Path) -> Path:
    image_path = Path(image)
    if image_path.is_absolute():
        return image_path
    return image_root / image_path


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    model, image_size = load_model(args.checkpoint, tokenizer, device)

    total = 0
    correct = 0
    by_task: dict[str, dict[str, int]] = {}
    out_f = args.output_jsonl.open("w", encoding="utf-8") if args.output_jsonl else None

    try:
        with args.test_jsonl.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if args.limit and idx >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)
                image_path = resolve_image_path(sample["image"], args.image_root)
                img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
                arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

                question = normalize_text(sample["question"])
                label_space = normalize_label(sample.get("label_space", "yes_no"))
                task_type = normalize_label(sample.get("task_type", "unknown"))
                prompt = build_prompt(question, label_space)
                raw_pred = model.generate(
                    arr,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                pred_text = extract_answer(raw_pred, label_space)
                gt_text = normalize_label(sample["answer"])

                total += 1
                is_correct = pred_text == gt_text
                correct += int(is_correct)
                by_task.setdefault(task_type, {"total": 0, "correct": 0})
                by_task[task_type]["total"] += 1
                by_task[task_type]["correct"] += int(is_correct)

                row = {
                    "idx": idx,
                    "task_type": task_type,
                    "label_space": label_space,
                    "question": question,
                    "prompt": prompt,
                    "gt": gt_text,
                    "raw_pred": raw_pred,
                    "pred": pred_text,
                    "correct": is_correct,
                    "image": str(image_path),
                }
                print(json.dumps(row, ensure_ascii=False))
                if out_f:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out_f:
            out_f.close()

    print("\n=== summary ===")
    print(f"total={total} correct={correct} acc={(correct / total) if total else 0.0:.4f}")
    for task_type, stats in sorted(by_task.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"{task_type}: {stats['correct']}/{stats['total']} acc={acc:.4f}")


if __name__ == "__main__":
    main()
