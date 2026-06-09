from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def normalize_label(value: str) -> str:
    return " ".join(value.strip().split()).lower()


@dataclass(frozen=True)
class PPEExample:
    image: str
    question: str
    answer: str
    task_type: str
    label_space: str
    split: str


class PPEVQADataset(Dataset):
    """Dataset for JSONL rows produced by scripts/convert_ppe_yolo.py."""

    def __init__(self, jsonl_path: str | Path, image_root: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.rows: list[PPEExample] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj: dict[str, Any] = json.loads(line)
                self.rows.append(
                    PPEExample(
                        image=obj["image"],
                        question=obj["question"],
                        answer=normalize_label(str(obj["answer"])),
                        task_type=normalize_label(str(obj.get("task_type", "unknown"))),
                        label_space=normalize_label(str(obj.get("label_space", "yes_no"))),
                        split=normalize_label(str(obj.get("split", "train"))),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute():
            return p
        if self.image_root is not None:
            return self.image_root / p
        return (self.jsonl_path.parent / p).resolve()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = self._resolve_image_path(row.image)
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "question": row.question,
            "answer": row.answer,
            "task_type": row.task_type,
            "label_space": row.label_space,
            "split": row.split,
            "image_path": str(image_path),
        }
