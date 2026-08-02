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
chore: initialize DVC with Google Drive remote
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

## Data and DVC

The remote is a **shared Google Drive folder**, accessed through the Google Drive
for Desktop mount rather than the Drive API. There is no token and no service
account to configure -- if you can see the folder in Explorer/Finder, DVC can
use it.

It used to be a HuggingFace WebDAV remote. That is gone, and any HF token you
still have lying around should be revoked.

One-time setup per person:

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
   and sign in with the account the `dog-breed-ood-dvc` folder is shared with.
2. Wait for the folder to appear in your mount.
3. Point DVC at *your* mount path. The committed default is Ryan's Windows
   path, so everyone else overrides it locally:

```bash
# Windows (adjust the drive letter if yours differs)
dvc remote modify --local gdrive_remote url "G:/My Drive/dog-breed-ood-dvc"

# macOS
dvc remote modify --local gdrive_remote url \
    "/Volumes/GoogleDrive/My Drive/dog-breed-ood-dvc"

dvc pull
```

`--local` writes to `.dvc/config.local`, which is git-ignored -- so your machine's
path never ends up in a PR.

Everyday use: `dvc pull` to get data, `dvc push` after a pipeline run, and commit
the resulting `.dvc` / `dvc.lock` files.

**Be patient after a push.** DVC writes thousands of small files, and Drive for
Desktop syncs them in the background. `dvc push` returning is not the same as
the data being uploaded -- check the Drive icon in your system tray before
telling someone else to pull.

## What never goes in Git

- Any file in data/ (use DVC)
- Any .pt / .pth / .pkl model file (use DVC)
- mlruns/ or mlflow.db
- .env files or any file containing tokens or API keys
- .dvc/config.local
- .secrets/ or any service account .json key

If you accidentally commit any of these:
```bash
git rm --cached path/to/file
git commit -m "fix: remove accidentally committed file"
git push
```
