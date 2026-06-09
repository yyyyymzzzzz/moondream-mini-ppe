# Release Checklist

Before opening the GitHub repository:

- Replace `authors` in `pyproject.toml`.
- Add an accurate model card if you publish trained weights.
- Confirm every dataset source permits redistribution or keep datasets external.
- Confirm derived-code attribution and notices from upstream Moondream are preserved where required.
- Run Python import checks and at least one dry-run conversion on a tiny fixture.
- Run `npm run build` in `apps/web`.
- Add screenshots or a short demo GIF to the README if you want a friendlier landing page.
- Tag the first release as `v0.1.0`.
