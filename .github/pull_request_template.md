## What does this PR do?
<!-- One sentence describing what this PR adds or changes -->


## Related PR / depends on
<!-- List any PRs that must be merged before this one -->
- Depends on: #

## Type of change
- [ ] feat: new feature
- [ ] fix: bug fix
- [ ] chore: setup, config, dependencies
- [ ] docs: documentation only
- [ ] test: tests only
- [ ] refactor: code restructure, no behavior change

## Files changed
<!-- List the key files added or modified -->
-
-

## DVC pipeline changes
- [ ] No DVC changes in this PR
- [ ] dvc.yaml updated
- [ ] dvc.lock updated and committed
- [ ] dvc push completed before opening this PR

## Test results
```
# paste output of: python -m pytest tests/ -v
```

## Verify steps for reviewer
<!-- Exact commands reviewer should run to confirm this works -->
```bash

```

## Checklist before requesting review
- [ ] I staged specific files only (never git add .)
- [ ] Tests are on this branch alongside the feature
- [ ] No .pt / .pkl / data files committed (run: git check-ignore -v models/)
- [ ] No hardcoded hyperparameters in .py files (all values in params.yaml)
- [ ] dvc.lock committed if pipeline was run
- [ ] Commit messages follow conventional format (feat: / chore: / fix: / docs: / test:)
- [ ] Branch is up to date with dev (run: git merge dev before pushing)
