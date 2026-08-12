# Contributing Guide

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| New feature | feat/short-description | feat/data-pipeline |
| Bug fix | fix/short-description | fix/entropy-threshold |
| Setup/config | chore/short-description | chore/dvc-setup |
| Documentation | docs/short-description | docs/model-card |
| Tests only | test/short-description | test/api-edge-cases |

Always branch off develop, never off main.

```bash
git checkout develop
git pull origin develop
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
# 1. Make sure you are up to date with develop
git checkout develop && git pull origin develop
git checkout your-branch
git merge develop

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

- Every PR targets develop, never main directly
- Minimum 2 approvals required before merge
- At least one reviewer must be from a different area (data person reviews model PR, etc.)
- Fill in the PR template completely -- blank sections will not be approved
- Delete the branch after merge

## Merge strategy

Use Squash and merge for feature branches into develop.
Use Merge commit for the final develop into main PR.

## Data and DVC

The remote is a **shared Google Drive folder**, reached through the Google Drive
API. The URL in `.dvc/config` is `gdrive://<folder-id>`, which means the same
thing on every machine -- Windows or macOS, with or without Google Drive for
Desktop installed. You do not need to mount anything.

It used to be a HuggingFace WebDAV remote, and briefly a local mount path. Both
are gone. Any HF token you still have lying around should be revoked.

### One-time setup per person

You need two values from Ryan (posted in the group chat, never in this repo):
a **client ID** and a **client secret**. These belong to our own Google Cloud
OAuth app -- DVC's built-in one is blocked by Google.

```bash
pip install -r requirements.txt

dvc remote modify --local gdrive_remote gdrive_client_id "<CLIENT_ID>"
dvc remote modify --local gdrive_remote gdrive_client_secret "<CLIENT_SECRET>"

dvc pull
```

On the first `dvc pull` a browser opens. Sign in with **the Google account the
Drive folder was shared with** -- if you use a different one, you will get
"This app is blocked", because only listed test users are allowed. You will also
see an "unverified app" warning: click **Advanced -> Go to DVC dogbreed**. That
is expected for an app Google has not reviewed.

`--local` writes to `.dvc/config.local`, which is git-ignored. **Never commit
these credentials** -- this repository is public.

### Everyday use

`dvc pull` to get data, `dvc push` after a pipeline run, and commit the
resulting `.dvc` / `dvc.lock` files.

`dvc pull` alone is fine now that `models/resnet18_best.pt` has been pushed to
the remote. If you hit a missing-object error on a fresh clone, someone forgot
to `dvc push` after the last pipeline run, not a config problem.

### Gotchas

**Windows path length.** DVC's run cache stacks two 64-character hashes, which
can exceed Windows' 260-character `MAX_PATH` limit and make `dvc repro` fail
with `[Errno 2]`. Clone into a short path such as `C:\dev\`, not
`OneDrive\Desktop\...`. Or enable long paths:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
git config --global core.longpaths true
```

**Do not put the repo inside OneDrive or Google Drive.** Sync clients grab
`.git` files mid-write and corrupt them, and they will try to sync your entire
`.dvc/cache` -- tens of thousands of files. Keep the working copy on plain local
disk. The DVC remote is the backup; the working copy does not need syncing.

**Storage.** The Drive folder lives on a 15 GB account. The dataset is ~2.4 GB.
Keep an eye on headroom before pushing model checkpoints.

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