# Model Card: Moondream Mini PPE

## Model Summary

Moondream Mini PPE is a compact vision-language model for construction-site PPE visual question answering. It is designed for short, constrained answers instead of open-ended image description.

The model is implemented in `src/moondream_mini/model.py` as `MiniMoondream`.

## Intended Use

Primary use cases:

- PPE safety QA on construction-site images
- yes/no PPE existence checks
- coarse worker/PPE counting
- coarse left/center/right localization
- local experimentation with small VLM training pipelines

The model is not intended to replace a production safety system without additional validation, calibration, and human review.

## Architecture

The default mini configuration uses:

- patch-based image encoder
- compact Transformer vision blocks
- causal Transformer text decoder
- image tokens prepended before text tokens
- language-model head for answer generation

The final v6 checkpoint used causal self-attention inside its compact vision blocks. The `vision_is_causal` configuration defaults to `true` for checkpoint compatibility; changing it defines a new model variant that must be retrained and reevaluated.

The model generates short labels such as `yes`, `no`, `0`, `1`, `2`, `3+`, `left`, `center`, and `right`.

## Training Task

The training pipeline converts YOLO PPE detection labels into VQA rows. Example:

```text
Question: How many workers wear safety helmets?
Options: 0, 1, 2, 3+
Answer: 2
```

Prompt tokens are masked from the loss, and the model is optimized to generate the answer tokens.

## Dataset Optimization

The PPE-specific conversion strategy is part of the model recipe:

- converts detection boxes into language supervision
- uses fixed answer spaces for exact-match evaluation
- uses coarse count and location buckets for small-model robustness
- samples multiple task types from the same image
- keeps prompt templates short and stable

## Limitations

- The model is specialized for PPE QA and may not generalize to broad VLM tasks.
- Coarse labels are easier to train but less expressive than bounding boxes or dense captions.
- Performance depends strongly on dataset quality, camera angle, class balance, and tokenizer choice.
- The final checkpoint requires the exact `moondream_starmie_v1` tokenizer vocabulary.
- Published weights should include evaluation results for the exact dataset and split used.
