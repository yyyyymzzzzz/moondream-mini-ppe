# Experiments and Reproducibility

## Final v6 experiment

The reported Moondream-mini results come from the final v6 experiment completed on June 5-6, 2026:

- 496 test images and 1,488 VQA samples
- 647 yes/no, 679 count, and 162 location questions
- fine-tuned Moondream-mini: 1,073/1,488 correct (72.11%)
- base Moondream comparison: 1,221/1,488 correct (82.06%)
- final checkpoint: `moondream_mini_20260605-192733_best.pt`
- tokenizer: `moondream_starmie_v1`

The original Linux paths under `/home/ymz/Workspace/moondream/` were machine-specific. The commands below preserve the final hyperparameters while using this repository's script names and portable relative paths.

## Downloaded artifact layout

Download [Big-Data-Analysis-and-Application-Course-Project](https://pan.sjtu.edu.cn/web/share/554d112a8ad57d66a08c73cecf679225) from SJTU Cloud with extraction code `3gsy`. The cloud UI lists `raw–data`, `moondream–mini–v6–checkpoint`, and `moondream_ppe_vqa_data_v6`. Rename the typographic dashes to ASCII hyphens so the repository commands use portable shell paths:

```bash
mv 'raw–data' raw-data
mv 'moondream–mini–v6–checkpoint' moondream-mini-v6-checkpoint
```

The resulting repository-root layout is:

```text
raw-data/
moondream_ppe_vqa_data_v6/
moondream-mini-v6-checkpoint/
```

The checkpoint is only compatible with the original `moondream_starmie_v1/tokenizer.json`. Place it at `artifacts/moondream_starmie_v1/tokenizer.json`. If that tokenizer is not inside the cloud bundle, add it before describing the bundle as independently reproducible.

## Repository-aligned commands

### Dataset conversion

```bash
python scripts/convert_ppe_yolo.py \
  --data-dir raw-data \
  --output-dir moondream_ppe_vqa_data_v6 \
  --seed 42 \
  --copy-images \
  --samples-per-image 3
```

### Training

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

The historical command did not record a training seed. The repository command adds seed `42` for deterministic future runs; it does not claim bit-for-bit reproduction of the original optimization trajectory.

### Evaluation

```bash
python scripts/evaluate.py \
  --test-jsonl moondream_ppe_vqa_data_v6/test.jsonl \
  --image-root moondream_ppe_vqa_data_v6 \
  --tokenizer artifacts/moondream_starmie_v1 \
  --checkpoint moondream-mini-v6-checkpoint/moondream_mini_20260605-192733_best.pt \
  --max-new-tokens 8 \
  --temperature 0.0 \
  --repetition-penalty 1.08 \
  --no-repeat-ngram-size 3 \
  --output-jsonl artifacts/evaluation/moondream-mini-v6-test-predictions.jsonl
```

The model configuration retains the causal vision-block attention behavior used to train the final v6 checkpoint. New experiments may set `vision_is_causal=false`, but metrics from that variant must not be presented as the final v6 result without retraining and reevaluation.

## Provenance checklist

Record the following with every reproduced result:

- Git commit
- dataset conversion report
- tokenizer SHA-256 checksum
- checkpoint SHA-256 checksum
- Python, PyTorch and device versions
- per-example prediction JSONL and aggregate metrics

The full base Moondream baseline evaluator is not part of this repository. Its values should be treated as reported comparison results rather than an automated check produced by the open-source code.
