# Course Submission Guide

## Source-code package

Do not compress the working directory directly. It contains ignored datasets, environments, logs, frontend dependencies, and local configuration that are not part of the source submission.

Verify the source, review and commit all intended changes, then create a clean archive from `HEAD`:

```bash
python scripts/check_submission.py
git status
# Commit the intended files before continuing; git archive excludes uncommitted changes.
git archive --format=zip --output ../moondream-mini-ppe-course.zip HEAD
```

The source archive intentionally excludes `.env`, datasets, checkpoints, tokenizer files, logs, `node_modules`, caches, and generated output.

## Offline demonstration package

If grading requires the program to run without downloading or training, prepare a separate authorized bundle containing:

- `artifacts/tokenizer/tokenizer.json`
- a final `checkpoints/**/_best.pt` checkpoint
- a small set of redistributable example images
- `.env.example` copied to `.env`

Then run:

```bash
python scripts/check_submission.py --require-demo-assets
./run_demo.sh
```

Do not include the entire third-party dataset unless its license explicitly permits redistribution. Include source URLs and license information for every bundled sample image.

The local smoke checkpoint is only a pipeline test and must not be presented as the final trained model.
