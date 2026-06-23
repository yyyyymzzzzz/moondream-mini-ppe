#!/usr/bin/env python3
"""Check that a source archive is clean and optionally ready for an offline demo."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "apps/api/app.py",
    "apps/web/package-lock.json",
    "scripts/train.py",
    "scripts/infer.py",
    "scripts/evaluate.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-demo-assets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")

    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if "Your Name" in metadata:
        errors.append("pyproject.toml still contains the author placeholder")

    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()
    if status:
        print("NOTICE: the working tree contains uncommitted changes:")
        print(status)

    if args.require_demo_assets:
        tokenizer = ROOT / "artifacts/tokenizer/tokenizer.json"
        checkpoints = list((ROOT / "checkpoints").glob("**/*_best.pt")) if (ROOT / "checkpoints").exists() else []
        if not tokenizer.is_file():
            errors.append("offline demo tokenizer is missing")
        if not checkpoints:
            errors.append("offline demo best checkpoint is missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Submission source checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
