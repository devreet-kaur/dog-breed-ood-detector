# Contributing Guide

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| New feature | feat/short-description | feat/data-pipeline |
| Bug fix | fix/short-description | fix/entropy-threshold |
| Setup/config | chore/short-description | chore/dvc-setup |
| Documentation | docs/short-description | docs/model-card |
| Tests only | test/short-description | test/api-edge-cases |

Always branch off dev, never off main.

```bash
git checkout dev
git pull origin dev
git checkout -b feat/your-task-name
```

## Commit message format

Every commit must follow conventional commits:

```
type: short description in lowercase

Examples:
feat: add bounding box crop to prepare.py
chore: initialize DVC with HuggingFace remote
fix: correct entropy threshold calibration
docs: update model card with final metrics
test: add edge case tests for invalid image input
```

Never write vague messages like "fix bug", "update", or "changes".

## Before opening a PR

Run all of these and confirm they pass:

```bash
# 1. Make sure you are up to date with dev
git checkout dev && git pull origin dev
git checkout your-branch
git merge dev

# 2. Run tests
python -m pytest tests/ -v

# 3. Check no data or model files are staged
git status
git check-ignore -v models/
git check-ignore -v data/

# 4. If you ran the pipeline, push DVC outputs and commit the lock
dvc push
git add dvc.lock
git commit -m "chore: update dvc.lock after pipeline run"

# 5. Lint
pip install ruff
ruff check src/ tests/
```

## PR rules

- Every PR targets dev, never main directly
- Minimum 2 approvals required before merge
- At least one reviewer must be from a different area (data person reviews model PR, etc.)
- Fill in the PR template completely -- blank sections will not be approved
- Delete the branch after merge

## Merge strategy

Use Squash and merge for feature branches into dev.
Use Merge commit for the final dev into main PR (#10).

## What never goes in Git

- Any file in data/ (use DVC)
- Any .pt / .pth / .pkl model file (use DVC)
- mlruns/ or mlflow.db
- .env files or any file containing tokens or API keys
- .dvc/config.local

If you accidentally commit any of these:
```bash
git rm --cached path/to/file
git commit -m "fix: remove accidentally committed file"
git push
```
