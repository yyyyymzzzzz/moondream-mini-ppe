#!/usr/bin/env python3
"""Convert PPE YOLO data into a classification-style Moondream dataset.

v5 goals:
- natural-language input questions
- fixed output labels from small class sets
- classification-style evaluation via exact label choice
- stable question templates with explicit options

Output JSONL format:
{
  "image": "images/train/xxx.jpg",
  "question": "...",
  "answer": "yes|no|0|1|2|3+|left|center|right",
  "task_type": "yes_no|count|location",
  "label_space": "yes_no|count_4|location_3",
  "split": "train|val|test"
}

Recommended usage:
    python scripts/convert_ppe_yolo.py \
        --data-dir /path/to/data \
        --output-dir moondream_ppe_vqa_v5 \
        --seed 42 \
        --copy-images
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

CLASS_NAME_ALIASES = {
    "no_helmet": "no-helmet",
    "no_vest": "no-vest",
    "no-helmet": "no-helmet",
    "no-vest": "no-vest",
}

DEFAULT_CLASS_NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]

# Very small template set on purpose.
QUESTION_TEMPLATES = {
    "count_person": ["How many workers are in the image?"],
    "count_helmet": ["How many workers wear safety helmets?"],
    "count_vest": ["How many workers wear safety vests?"],
    "yes_no_helmet": ["Is any worker wearing a safety helmet?"],
    "yes_no_vest": ["Is any worker wearing a safety vest?"],
    "existence_person": ["Is there at least one worker in the image?"],
    "location_helmet": ["Where is the helmet-wearing worker?"],
    "location_vest": ["Where is the vest-wearing worker?"],
}

LOCATION_BUCKETS = {
    (0, 0): "left",
    (1, 0): "center",
    (2, 0): "right",
    (0, 1): "left",
    (1, 1): "center",
    (2, 1): "right",
    (0, 2): "left",
    (1, 2): "center",
    (2, 2): "right",
}

@dataclass(frozen=True)
class BoxRecord:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Counts:
    person: int
    helmet: int
    vest: int
    no_helmet: int
    no_vest: int


YES_NO_OPTIONS = ["yes", "no"]
COUNT_OPTIONS = ["0", "1", "2", "3+"]
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing one or more YOLO dataset roots")
    p.add_argument("--output-dir", type=Path, default=Path("moondream_ppe_vqa_v3"))
    p.add_argument("--split-map", type=str, default="train:train,val:valid,test:test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("--absolute-image-paths", action="store_true")
    p.add_argument(
        "--samples-per-image",
        type=int,
        default=3,
        help="How many QA pairs to keep per image (after prioritization)",
    )
    return p.parse_args()


def normalize_class_name(name: str) -> str:
    return CLASS_NAME_ALIASES.get(name, name.replace("_", "-").strip())


def load_class_names(dataset_root: Path) -> list[str]:
    with (dataset_root / "data.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    names = config.get("names") or DEFAULT_CLASS_NAMES
    return [normalize_class_name(str(n)) for n in names]


def discover_dataset_roots(data_dir: Path) -> list[Path]:
    roots = sorted([p for p in data_dir.iterdir() if p.is_dir() and (p / "data.yaml").exists()], key=lambda p: p.name)
    if not roots:
        raise ValueError(f"No YOLO dataset roots with data.yaml found under {data_dir}")
    return roots


def parse_split_map(split_map: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in split_map.split(","):
        out_split, src_split = item.split(":", 1)
        mapping[out_split.strip()] = src_split.strip()
    return mapping


def resolve_split_dir(dataset_root: Path, split_name: str) -> Path:
    candidates = [split_name]
    if split_name == "val":
        candidates.append("valid")
    elif split_name == "valid":
        candidates.append("val")
    for candidate in candidates:
        p = dataset_root / candidate
        if p.exists():
            return p
    return dataset_root / split_name


def image_dir_and_label_dir(dataset_root: Path, split_name: str) -> tuple[Path, Path]:
    split_dir = resolve_split_dir(dataset_root, split_name)
    return split_dir / "images", split_dir / "labels"


def iter_images(image_dir: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        yield from sorted(image_dir.glob(ext))


def read_labels(label_path: Path) -> list[BoxRecord]:
    records: list[BoxRecord] = []
    if not label_path.exists():
        return records
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            records.append(
                BoxRecord(
                    class_id=int(float(parts[0])),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                )
            )
    return records


def count_objects(records: list[BoxRecord], class_names: list[str]) -> Counts:
    counts = {normalize_class_name(n): 0 for n in class_names}
    for rec in records:
        if 0 <= rec.class_id < len(class_names):
            counts[normalize_class_name(class_names[rec.class_id])] += 1
    return Counts(
        person=counts.get("person", 0),
        helmet=counts.get("helmet", 0),
        vest=counts.get("vest", 0),
        no_helmet=counts.get("no-helmet", 0),
        no_vest=counts.get("no-vest", 0),
    )


def bucket_location(x: float, y: float) -> str:
    x_idx = 0 if x < 1 / 3 else 2 if x > 2 / 3 else 1
    y_idx = 0 if y < 1 / 3 else 2 if y > 2 / 3 else 1
    return LOCATION_BUCKETS[(x_idx, y_idx)]


def answer_yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def build_count_samples(counts: Counts) -> list[tuple[str, str, str]]:
    return [
        (QUESTION_TEMPLATES["count_person"][0], str(counts.person), "count"),
        (QUESTION_TEMPLATES["count_helmet"][0], str(counts.helmet), "count"),
        (QUESTION_TEMPLATES["count_vest"][0], str(counts.vest), "count"),
    ]


def build_yes_no_samples(counts: Counts) -> list[tuple[str, str, str]]:
    return [
        (QUESTION_TEMPLATES["yes_no_helmet"][0], answer_yes_no(counts.helmet > 0), "yes_no"),
        (QUESTION_TEMPLATES["yes_no_vest"][0], answer_yes_no(counts.vest > 0), "yes_no"),
        (QUESTION_TEMPLATES["existence_person"][0], answer_yes_no(counts.person > 0), "yes_no"),
    ]


def classify_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def classify_location(x: float, y: float) -> str:
    return bucket_location(x, y)


def build_easy_samples(
    records: list[BoxRecord], counts: Counts, class_names: list[str]
) -> list[tuple[str, str, str, str]]:
    # Classification-style dataset: each answer belongs to a fixed label space.
    samples: list[tuple[str, str, str, str]] = []
    samples.append((QUESTION_TEMPLATES["yes_no_helmet"][0], answer_yes_no(counts.helmet > 0), "yes_no", "yes_no"))
    samples.append((QUESTION_TEMPLATES["yes_no_vest"][0], answer_yes_no(counts.vest > 0), "yes_no", "yes_no"))
    samples.append((QUESTION_TEMPLATES["existence_person"][0], answer_yes_no(counts.person > 0), "yes_no", "yes_no"))
    samples.append((QUESTION_TEMPLATES["count_person"][0], classify_count(counts.person), "count", "count_4"))
    samples.append((QUESTION_TEMPLATES["count_helmet"][0], classify_count(counts.helmet), "count", "count_4"))
    samples.append((QUESTION_TEMPLATES["count_vest"][0], classify_count(counts.vest), "count", "count_4"))
    first_locations: dict[str, BoxRecord] = {}
    for rec in records:
        if 0 <= rec.class_id < len(class_names):
            cls_name = normalize_class_name(class_names[rec.class_id])
            if cls_name in {"helmet", "vest"} and cls_name not in first_locations:
                first_locations[cls_name] = rec
    for cls_name, template_name in (("helmet", "location_helmet"), ("vest", "location_vest")):
        if rec := first_locations.get(cls_name):
            samples.append(
                (
                    QUESTION_TEMPLATES[template_name][0],
                    classify_location(rec.x_center, rec.y_center),
                    "location",
                    "location_3",
                )
            )
    return samples


def copy_or_reference_image(
    image_path: Path,
    dataset_root: Path,
    out_split: str,
    output_dir: Path,
    copy_images: bool,
    absolute_paths: bool,
) -> str:
    if absolute_paths:
        return str(image_path.resolve())
    if copy_images:
        out_dir = output_dir / "images" / out_split
        out_dir.mkdir(parents=True, exist_ok=True)
        copied_name = f"{dataset_root.name}_{image_path.name}"
        copied_path = out_dir / copied_name
        if not copied_path.exists():
            shutil.copy2(image_path, copied_path)
        return str(Path("images") / out_split / copied_name)
    # Without copied images, use an absolute reference so the generated JSONL
    # remains usable when it is stored outside the source dataset root.
    return str(image_path.resolve())


def process_split(
    split_name: str,
    source_split: str,
    dataset_roots: list[Path],
    class_names: list[str],
    output_dir: Path,
    rng: random.Random,
    samples_per_image: int,
    copy_images: bool,
    absolute_image_paths: bool,
) -> dict[str, int]:
    out_path = output_dir / f"{split_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "samples": 0, "yes_no": 0, "count": 0, "location": 0}
    seen: set[str] = set()

    with out_path.open("w", encoding="utf-8") as f:
        for dataset_root in dataset_roots:
            image_dir, label_dir = image_dir_and_label_dir(dataset_root, source_split)
            for image_path in iter_images(image_dir):
                key = f"{dataset_root.name}:{image_path.name}"
                if key in seen:
                    continue
                seen.add(key)
                stats["images"] += 1
                label_path = label_dir / f"{image_path.stem}.txt"
                records = read_labels(label_path)
                counts = count_objects(records, class_names)

                # Stable classification samples with explicit label spaces.
                samples = build_easy_samples(records, counts, class_names)
                rng.shuffle(samples)
                samples = samples[:samples_per_image]

                image_value = copy_or_reference_image(
                    image_path,
                    dataset_root,
                    split_name,
                    output_dir,
                    copy_images,
                    absolute_image_paths,
                )
                for question, answer, task_type, label_space in samples:
                    f.write(
                        json.dumps(
                            {
                                "image": image_value,
                                "question": question,
                                "answer": answer,
                                "task_type": task_type,
                                "label_space": label_space,
                                "split": split_name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["samples"] += 1
                    stats[task_type] += 1

    return stats


def write_report(output_dir: Path, class_names: list[str], split_stats: dict[str, dict[str, int]]) -> None:
    report = {
        "class_names": class_names,
        "split_stats": split_stats,
        "notes": "v3 uses fewer question templates and shorter answers.",
    }
    with (output_dir / "conversion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots = discover_dataset_roots(args.data_dir)
    reference_class_names = load_class_names(dataset_roots[0])
    for idx, root in enumerate(dataset_roots[1:], start=2):
        class_names = load_class_names(root)
        if class_names != reference_class_names:
            raise ValueError(f"Class mismatch in dataset {idx}: {class_names} != {reference_class_names}")

    split_map = parse_split_map(args.split_map)
    split_stats = {}
    for out_split, src_split in split_map.items():
        split_stats[out_split] = process_split(
            split_name=out_split,
            source_split=src_split,
            dataset_roots=dataset_roots,
            class_names=reference_class_names,
            output_dir=output_dir,
            rng=rng,
            samples_per_image=args.samples_per_image,
            copy_images=args.copy_images,
            absolute_image_paths=args.absolute_image_paths,
        )

    write_report(output_dir, reference_class_names, split_stats)
    print("Conversion complete.")
    for split_name, stats in split_stats.items():
        print(
            f"{split_name}: {stats['images']} images -> {stats['samples']} samples "
            f"(count={stats['count']}, yes_no={stats['yes_no']})"
        )
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
