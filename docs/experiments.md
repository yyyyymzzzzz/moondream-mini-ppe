# Experiments and Reproducibility

## Reported evaluation

The headline results in the README come from the original course-project evaluation on the converted test split:

- 496 images
- 1,488 VQA samples
- 647 yes/no, 679 count, and 162 location questions
- fine-tuned Moondream-mini: 1,073/1,488 correct (72.11%)
- base Moondream comparison: 1,221/1,488 correct (82.06%)

These values are a report snapshot. The final checkpoint and per-example prediction JSONL are not currently distributed in the public repository, so the headline metrics cannot be independently reproduced from a clean clone alone. Do not describe the public repository as fully reproducible until those artifacts or stable download links and checksums are added.

## Evaluation command

For the mini model, use:

```bash
python scripts/evaluate.py \
  --test-jsonl data/moondream_ppe_vqa/test.jsonl \
  --image-root data/moondream_ppe_vqa \
  --tokenizer artifacts/tokenizer \
  --checkpoint checkpoints/moondream-mini/moondream_mini_best.pt \
  --output-jsonl artifacts/evaluation/moondream-mini-test-predictions.jsonl
```

Record the following with every result:

- Git commit
- dataset source/version and conversion report
- tokenizer checksum
- checkpoint checksum
- full training command and random seed
- PyTorch/Python versions and device
- prediction JSONL and aggregate metrics

Training now defaults to seed `42` and stores the seed, model configuration, and training arguments in each checkpoint.

## Baseline status

The repository does not currently contain an executable evaluator for the full base Moondream baseline. The baseline values in the README should therefore be treated as reported comparison results, not as an automated test produced by this codebase.
