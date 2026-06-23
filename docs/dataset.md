# Dataset Format

The dataset converter is part of the model design. It adapts standard PPE object detection annotations into a constrained VQA format that a small VLM can learn efficiently.

The converter writes one JSON object per line:

```json
{
  "image": "images/train/dataset_a_frame_001.jpg",
  "question": "How many workers wear safety helmets?",
  "answer": "2",
  "task_type": "count",
  "label_space": "count_4",
  "split": "train"
}
```

## Supported Label Spaces

`yes_no`

- `yes`
- `no`

`count_4`

- `0`
- `1`
- `2`
- `3+`

`location_3`

- `left`
- `center`
- `right`

## Conversion Algorithm

`scripts/convert_ppe_yolo.py` reads YOLO labels and derives several QA samples from each image:

- helmet existence: `Is any worker wearing a safety helmet?`
- vest existence: `Is any worker wearing a safety vest?`
- worker existence: `Is there at least one worker in the image?`
- worker count: `How many workers are in the image?`
- helmet count: `How many workers wear safety helmets?`
- vest count: `How many workers wear safety vests?`
- helmet location: `Where is the helmet-wearing worker?`
- vest location: `Where is the vest-wearing worker?`

The converter then shuffles candidate samples and keeps `--samples-per-image` rows. This prevents a single image from producing too many near-duplicate examples while still mixing task types across the dataset.

## Why These Labels

The label spaces are deliberately coarse:

- `yes_no` handles PPE existence and worker presence.
- `count_4` caps counts at `3+`, which keeps the task robust when there are many people or partially visible objects.
- `location_3` uses left/center/right instead of nine-grid or bounding-box outputs because coarse spatial answers are easier for a mini VLM to learn and evaluate.

The prompt always includes explicit options. This makes training and inference match:

```text
<image> Question: Is any worker wearing a safety vest? Options: yes, no Answer:
<image> Question: How many workers are in the image? Options: 0, 1, 2, 3+ Answer:
<image> Question: Where is the helmet-wearing worker? Options: left, center, right Answer:
```

## Notes

The converter expects YOLO labels with normalized bounding boxes:

```text
class_id x_center y_center width height
```

Default class names are:

```text
helmet, no-helmet, no-vest, person, vest
```

If your dataset uses another order, make sure every dataset root has a correct `data.yaml`.

`--split-map` maps output names to source directories, for example `test:valid`. With `--copy-images`, the generated dataset remains portable. Without it, JSONL rows contain absolute paths back to the source images and are suitable only on the machine where conversion was run.
