# Local Mac Runbook

This guide verifies the open-source project locally on macOS using the two YOLO datasets from the original `moondream/data` directory.

Assumptions:

- Original repo: `~/Workspace/moondream`
- Open-source repo: `~/Workspace/moondream-mini-ppe`
- YOLO datasets:
  - `~/Workspace/moondream/data/construction safety.yolov8`
  - `~/Workspace/moondream/data/Personal-Protective-Equipment (PPE) Dataset`

## 1. Create Environment

```bash
conda create -y -n moondream-test python=3.10
conda activate moondream-test
cd ~/Workspace/moondream-mini-ppe
pip install -e .
```

For Apple Silicon, install PyTorch from the official PyTorch selector if the default wheel does not enable MPS correctly.

Check imports:

```bash
python - <<'PY'
import torch
from moondream_mini import MiniConfig, MiniMoondream
print("torch:", torch.__version__)
print("mps:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print(MiniMoondream(MiniConfig(vocab_size=128)))
PY
```

## 2. Convert YOLO Data

The converter discovers all dataset roots under `~/Workspace/moondream/data` that contain `data.yaml`.

```bash
cd ~/Workspace/moondream-mini-ppe
python scripts/convert_ppe_yolo.py \
  --data-dir ~/Workspace/moondream/data \
  --output-dir data/moondream_ppe_vqa \
  --copy-images \
  --samples-per-image 3
```

Inspect the output:

```bash
wc -l data/moondream_ppe_vqa/*.jsonl
head -3 data/moondream_ppe_vqa/train.jsonl
cat data/moondream_ppe_vqa/conversion_report.json
```

## 3. Prepare a Minimal Tokenizer

If you already have a tokenizer directory, put it at `artifacts/tokenizer/tokenizer.json`.

For a quick smoke test, train a tiny tokenizer from the generated JSONL text:

```bash
mkdir -p artifacts/tokenizer
python - <<'PY'
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

root = Path("data/moondream_ppe_vqa")
texts = []
for path in root.glob("*.jsonl"):
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        texts.append(f"<image> Question: {row['question']} Options: yes, no, 0, 1, 2, 3+, left, center, right Answer: {row['answer']} </s>")

tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(special_tokens=["<pad>", "<unk>", "</s>", "<image>"])
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save("artifacts/tokenizer/tokenizer.json")
print("vocab_size:", tokenizer.get_vocab_size())
PY
```

## 4. Smoke Train

This is only to verify the full pipeline. It is intentionally tiny.

```bash
python scripts/train.py \
  --train-jsonl data/moondream_ppe_vqa/train.jsonl \
  --val-jsonl data/moondream_ppe_vqa/val.jsonl \
  --image-root data/moondream_ppe_vqa \
  --tokenizer artifacts/tokenizer \
  --output-dir checkpoints/moondream-mini-smoke \
  --run-name smoke \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --image-size 160 \
  --max-text-len 96 \
  --device mps
```

If MPS is unavailable or unstable:

```bash
python scripts/train.py \
  --train-jsonl data/moondream_ppe_vqa/train.jsonl \
  --val-jsonl data/moondream_ppe_vqa/val.jsonl \
  --image-root data/moondream_ppe_vqa \
  --tokenizer artifacts/tokenizer \
  --output-dir checkpoints/moondream-mini-smoke \
  --run-name smoke \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --image-size 160 \
  --max-text-len 96 \
  --device cpu
```

## 5. CLI Inference

Pick one converted validation image:

```bash
IMAGE="$(find data/moondream_ppe_vqa/images/val -type f | head -1)"
CHECKPOINT="$(ls checkpoints/moondream-mini-smoke/*_best.pt | head -1)"

python scripts/infer.py \
  --image "$IMAGE" \
  --question "Is any worker wearing a safety helmet?" \
  --tokenizer artifacts/tokenizer \
  --checkpoint "$CHECKPOINT" \
  --device mps
```

Use `--device cpu` if needed.

## 6. API Check

Optional evaluation on the generated test split:

```bash
python scripts/evaluate.py \
  --test-jsonl data/moondream_ppe_vqa/test.jsonl \
  --image-root data/moondream_ppe_vqa \
  --tokenizer artifacts/tokenizer \
  --checkpoint "$CHECKPOINT" \
  --device mps \
  --limit 20
```

The recommended path is to use the unified startup script:

```bash
cat > .env <<EOF
MOONDREAM_CHECKPOINT=$CHECKPOINT
MOONDREAM_TOKENIZER=artifacts/tokenizer
MOONDREAM_DATA_ROOT=data/moondream_ppe_vqa
MOONDREAM_DEVICE=
MOONDREAM_CORS_ORIGINS=http://localhost:5173
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5173
EOF

./run_demo.sh
```

Manual backend check:

```bash
export MOONDREAM_CHECKPOINT="$CHECKPOINT"
export MOONDREAM_TOKENIZER=artifacts/tokenizer
export MOONDREAM_DATA_ROOT=data/moondream_ppe_vqa
uvicorn apps.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/gallery | head
```

## 7. Web Demo

```bash
cd ~/Workspace/moondream-mini-ppe/apps/web
npm install
npm run dev
```

Open `http://localhost:5173`.

## Notes

- Final v6 artifacts are available from [SJTU Cloud](https://pan.sjtu.edu.cn/web/share/554d112a8ad57d66a08c73cecf679225), extraction code `3gsy`. See `docs/experiments.md` for the exact final commands.
- This runbook skips downloading or comparing against the full base Moondream model.
- The smoke checkpoint is only for pipeline verification, not final accuracy.
- For a real run, use a stronger tokenizer, larger `image_size`, more epochs, and a held-out test evaluation.
- Before an offline course demonstration, run `python scripts/check_submission.py --require-demo-assets`; the public source archive does not contain the final checkpoint.
