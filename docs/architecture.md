# Architecture

## Python Package

`src/moondream_mini` contains reusable code:

- `model.py`: compact ViT-style image encoder, Transformer decoder, and image/text fusion
- `data.py`: JSONL PPE VQA dataset loader
- `prompts.py`: shared prompt construction, label-space inference, and answer extraction

Scripts and the API import from this package instead of reaching into app-specific paths.

## Mini Model Design

The project centers on `MiniMoondream`, a compact vision-language model designed for task-focused PPE QA rather than broad open-ended captioning.

Key classes:

- `MiniConfig`: model hyperparameters and sequence limits
- `VisionEncoder`: image patch encoder
- `MiniMoondream`: image-conditioned causal language model
- `TransformerBlock`: shared compact Transformer block

Default shape:

```text
image
  -> Conv2d patch embedding
  -> CLS token + learned positional embeddings
  -> 2-layer vision Transformer
  -> projection to text width
  -> image tokens

question tokens
  -> token embeddings
  -> text tokens

image tokens + text tokens
  -> causal Transformer decoder
  -> LM head
  -> short label answer
```

The fusion path is intentionally simple: encoded image tokens are prepended to text tokens. This keeps the model easy to inspect, easy to train from scratch, and small enough for local experimentation.

## Data Flow

```text
YOLO PPE dataset
  -> scripts/convert_ppe_yolo.py
  -> JSONL VQA rows + optional copied images
  -> scripts/train.py
  -> checkpoint
  -> scripts/infer.py or apps/api/app.py
  -> apps/web
```

## Model Shape

The mini model is intentionally small:

- image encoder: patch embedding + a few Transformer blocks
- text side: token embedding + causal Transformer decoder
- fusion: image tokens are prepended to text tokens
- output: language-model head over tokenizer vocabulary

Training treats PPE QA as short-answer generation over constrained label spaces. For each row, the prompt looks like:

```text
<image> Question: How many workers wear safety helmets? Options: 0, 1, 2, 3+ Answer:
```

The target is the short answer plus EOS. Prompt tokens are masked out of the loss, so optimization focuses on generating the answer.

## PPE Optimization Strategy

The model is small, so the project narrows the task instead of pretending the model is a general-purpose 2B VLM. The PPE pipeline helps in several ways:

- It converts object detection labels into QA supervision without manually writing conversations.
- It uses answer options in the prompt, reducing output entropy.
- It normalizes counts into `0`, `1`, `2`, and `3+`, which makes the label distribution less sparse.
- It turns object centers into coarse location answers, making localization learnable without coordinate decoding.
- It keeps templates short and stable, which is useful when training a compact decoder.
- It shares prompt construction between training, CLI inference, API inference, and the web UI.

This makes the repository a targeted small-model project: the important contribution is the combination of a lightweight VLM architecture and a constrained PPE VQA formulation.
