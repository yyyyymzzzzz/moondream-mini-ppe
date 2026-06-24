# Moondream Mini PPE

Moondream Mini PPE is a small, task-focused vision-language model built for construction-site PPE visual question answering. It keeps the Moondream-style idea of image-conditioned language generation, but uses a deliberately compact PyTorch architecture and a PPE-specific dataset conversion strategy so it can be trained and deployed locally on modest hardware.

This repository packages three parts into one open-source project:

- the custom `MiniMoondream` model (`src/moondream_mini/model.py`)
- a PPE-optimized YOLO-to-VQA converter (`scripts/convert_ppe_yolo.py`)
- training and inference scripts for constrained-answer PPE QA
- a FastAPI backend and React/Vite web demo (`apps/api`, `apps/web`)

The project is optimized around construction-site safety questions with fixed answer spaces:

- `yes_no`: `yes`, `no`
- `count_4`: `0`, `1`, `2`, `3+`
- `location_3`: `left`, `center`, `right`

This constrained formulation is intentional: instead of asking a tiny model to generate long free-form descriptions, the dataset and prompts turn PPE recognition into short, stable visual QA tasks that are easier to train, evaluate, and deploy.

## Contributions

This project is more than a web demo around an existing VLM. The main contributions are:

- **Moondream-mini**: a compact Moondream-style VLM implemented in PyTorch for low-cost local fine-tuning and inference.
- **Data-centric PPE adaptation**: a deterministic converter that turns YOLO bounding boxes into faithful VLM question-answer supervision.
- **Small-model task design**: constrained answer spaces and explicit prompt options reduce output ambiguity for a lightweight decoder.
- **Spatial discretization**: continuous object centers are mapped into human-readable location labels instead of asking the model to regress coordinates.
- **End-to-end system**: dataset conversion, training, CLI inference, FastAPI serving, and a React web interface are packaged together.

## Moondream-mini Model

The core model lives in [`src/moondream_mini/model.py`](src/moondream_mini/model.py). It is a lightweight reimplementation inspired by Moondream-style VLMs, not a copy of the full upstream model.

![Moondream-mini model architecture](docs/assets/model-architecture.svg)

Default mini configuration:

| Component | Default |
| --- | --- |
| Image size | `224` during training examples, configurable |
| Patch size | `16` |
| Vision width | `128` |
| Text width | `128` |
| Attention heads | `4` |
| Vision layers | `2` |
| Text layers | `3` |
| Feed-forward width | `384` |
| Fusion method | prepend image tokens before text tokens |

The model has three main parts:

- `VisionEncoder`: patch embedding, CLS token, positional interpolation, and compact Transformer blocks
- `MiniMoondream`: text token embeddings, image/text token type embeddings, causal decoder blocks, and LM head
- `generate`: short-answer autoregressive decoding with repetition control for stable label generation

## PPE-Specific Optimization

This repository is not only a generic mini VLM wrapper. The data and training path are adapted for PPE detection datasets:

- Converts YOLO boxes into natural-language QA pairs, so detection annotations can train a VLM-style interface.
- Uses small label spaces (`yes_no`, `count_4`, `location_3`) to reduce ambiguity and improve exact-match evaluation.
- Buckets object positions into left/center/right answers, which is more robust for small models than precise coordinate generation.
- Caps count answers at `3+`, avoiding brittle long-tail count classes.
- Samples multiple task types per image, mixing existence, counting, and coarse localization from the same annotation file.
- Uses prompt templates with explicit answer options, aligning training, CLI inference, API inference, and web demo behavior.

![PPE dataset conversion and training pipeline](docs/assets/ppe-pipeline.svg)

## Dataset Sources

The PPE VQA data used by this project is converted from public object-detection datasets:

- [Construction Safety Dataset on Roboflow Universe](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb)
- [Personal Protective Equipment (PPE) Dataset on Kaggle](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset)

For course-project reproduction, download the complete course bundle from [SJTU Cloud](https://pan.sjtu.edu.cn/web/share/554d112a8ad57d66a08c73cecf679225) with extraction code `3gsy`.

```text
Big-Data-Analysis-and-Application-Course-Project/
├── raw–data/                         # Original YOLO datasets
├── moondream–mini–v6–checkpoint/     # Final v6 training checkpoint
└── moondream_ppe_vqa_data_v6/        # Converted train/val/test JSONL and images
```

The cloud UI uses typographic dashes (`–`) in two directory names. For reliable shell commands, rename them to the repository's ASCII path convention after downloading:

```bash
mv 'raw–data' raw-data
mv 'moondream–mini–v6–checkpoint' moondream-mini-v6-checkpoint
```

The final v6 checkpoint must be used with the exact `moondream_starmie_v1/tokenizer.json` tokenizer from the original run. Place that tokenizer at `artifacts/moondream_starmie_v1/`. A tokenizer with another vocabulary size is not checkpoint-compatible. If the downloaded checkpoint folder does not contain the tokenizer, it still needs to be added to the shared bundle.

The Git repository itself does not contain these datasets. Before using or redistributing data from either the original sources or the course mirror, check the corresponding licenses and terms. To prepare the VQA data from the original YOLO datasets, run the converter locally.

## Evaluation Results

In the original project report, a fine-tuned Moondream-mini checkpoint was evaluated on a held-out PPE test set of 1,488 VQA samples. See [experiment provenance and reproduction notes](docs/experiments.md) for the exact command and current artifact limitations.

| Model | Task | Samples | Correct | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Fine-tuned Moondream-mini | Total | 1,488 | 1,073 | 72.11% |
| Fine-tuned Moondream-mini | Count | 679 | 421 | 62.00% |
| Fine-tuned Moondream-mini | Location | 162 | 102 | 62.96% |
| Fine-tuned Moondream-mini | Yes/No | 647 | 550 | 85.01% |
| Base Moondream | Total | 1,488 | 1,221 | 82.06% |
| Base Moondream | Count | 679 | 512 | 75.41% |
| Base Moondream | Location | 162 | 117 | 72.22% |
| Base Moondream | Yes/No | 647 | 592 | 91.50% |

The base Moondream model is stronger, as expected from a larger general-purpose VLM. The fine-tuned Moondream-mini result shows the tradeoff explored by this project: a much smaller, locally trainable model can still reach useful PPE QA performance, especially on the safety-critical yes/no compliance task. Count and coarse-location tasks remain harder for the mini model, but the constrained label design makes the gap measurable and gives a clear path for future improvements.

## Project Layout

```text
.
├── apps/
│   ├── api/                 # FastAPI inference service
│   └── web/                 # React/Vite visual demo
├── configs/                 # Training/model config examples
├── docs/                    # Architecture, dataset, model card, and release notes
├── scripts/
│   ├── convert_ppe_yolo.py  # YOLO PPE dataset to JSONL VQA
│   ├── train.py             # Mini VLM training entrypoint
│   ├── infer.py             # CLI inference
│   └── evaluate.py          # Test-set evaluation
└── src/moondream_mini/      # Reusable Python package
```

More details:

- [Model card](docs/model_card.md)
- [Architecture notes](docs/architecture.md)
- [Dataset conversion notes](docs/dataset.md)
- [Experiment and reproducibility notes](docs/experiments.md)
- [Local Mac runbook](docs/local_run_mac.md)
- [Course submission guide](SUBMISSION.md)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install the frontend separately:

```bash
cd apps/web
npm install
```

## Prepare PPE Data

Expected input is one or more YOLO dataset roots under a parent directory:

```text
/path/to/ppe-yolo/
└── dataset-a/
    ├── data.yaml
    ├── train/images
    ├── train/labels
    ├── valid/images
    └── valid/labels
```

After placing the downloaded `raw-data/` directory in the repository root, reproduce the v6 conversion with:

```bash
python scripts/convert_ppe_yolo.py \
  --data-dir raw-data \
  --output-dir moondream_ppe_vqa_data_v6 \
  --seed 42 \
  --copy-images \
  --samples-per-image 3
```

## Train

Place the final tokenizer at `artifacts/moondream_starmie_v1/tokenizer.json`, then run the repository-aligned version of the final training command:

```bash
python scripts/train.py \
  --train-jsonl moondream_ppe_vqa_data_v6/train.jsonl \
  --val-jsonl moondream_ppe_vqa_data_v6/val.jsonl \
  --image-root moondream_ppe_vqa_data_v6 \
  --output-dir moondream-mini-v6-checkpoint \
  --tokenizer artifacts/moondream_starmie_v1 \
  --epochs 30 \
  --batch-size 32 \
  --lr 3e-5 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --image-size 224 \
  --max-text-len 128 \
  --freeze-vision \
  --freeze-epochs 3 \
  --seed 42
```

## CLI Inference

```bash
python scripts/infer.py \
  --image moondream_ppe_vqa_data_v6/images/val/example.jpg \
  --question "How many workers wear safety helmets?" \
  --tokenizer artifacts/moondream_starmie_v1 \
  --checkpoint moondream-mini-v6-checkpoint/moondream_mini_20260605-192733_best.pt \
  --max-new-tokens 8
```

## Evaluate

```bash
python scripts/evaluate.py \
  --test-jsonl moondream_ppe_vqa_data_v6/test.jsonl \
  --image-root moondream_ppe_vqa_data_v6 \
  --tokenizer artifacts/moondream_starmie_v1 \
  --checkpoint moondream-mini-v6-checkpoint/moondream_mini_20260605-192733_best.pt \
  --max-new-tokens 8 \
  --temperature 0.0 \
  --repetition-penalty 1.08 \
  --no-repeat-ngram-size 3
```

## API and Web Demo

Create `.env` from `.env.example`, then point it at your local checkpoint and tokenizer:

```bash
cp .env.example .env
```

Edit `.env` if your checkpoint path is different, then start both backend and frontend:

```bash
./run_demo.sh
```

Open `http://localhost:5173`.

![Moondream Mini PPE web demo](docs/assets/web-demo.png)

Manual startup is also supported:

```bash
export MOONDREAM_CHECKPOINT=moondream-mini-v6-checkpoint/moondream_mini_20260605-192733_best.pt
export MOONDREAM_TOKENIZER=artifacts/moondream_starmie_v1
export MOONDREAM_DATA_ROOT=moondream_ppe_vqa_data_v6
uvicorn apps.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd apps/web
npm run dev
```

## Verify

Run the automated checks before submitting or publishing:

```bash
python -m unittest discover -s tests -v
python scripts/check_submission.py
cd apps/web && npm run build
```

To additionally require local tokenizer and best-checkpoint assets for an offline demonstration:

```bash
python scripts/check_submission.py --require-demo-assets
```

## What Is Not Committed

The repository intentionally excludes:

- downloaded `raw-data/`
- converted `moondream_ppe_vqa_data_v6/`
- downloaded `moondream-mini-v6-checkpoint/`
- tokenizer artifacts
- local logs, notebooks, cache files, and `.env`

The course artifact bundle is published separately through the SJTU Cloud link above so the Git history remains small.

## License

Apache-2.0. Check upstream model and dataset licenses before publishing derived weights or bundled sample data.
