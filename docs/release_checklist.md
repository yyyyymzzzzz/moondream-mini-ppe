# Release Checklist

Before opening the GitHub repository:

- Add author metadata to `pyproject.toml` if the release should publish a personal name.
- Add an accurate model card if you publish trained weights.
- Confirm every dataset source permits redistribution or keep datasets external.
- Confirm derived-code attribution and notices from upstream Moondream are preserved where required.
- Run Python import checks and at least one dry-run conversion on a tiny fixture.
- Run `python -m unittest discover -s tests -v`.
- Run `python scripts/check_submission.py` and, for an offline demo, add `--require-demo-assets`.
- Run `npm run build` in `apps/web`.
- Add screenshots or a short demo GIF to the README if you want a friendlier landing page.
- Tag the first release as `v0.1.0`.
