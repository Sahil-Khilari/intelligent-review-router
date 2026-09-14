# CI workflow

`ci.yml` runs the test suite on every pull request.

## Why it is not in `.github/workflows/`

Pushing a file under `.github/workflows/` requires the `workflow` scope on the
credential, which the token used here does not have:

```
refusing to allow a Personal Access Token to create or update workflow
`.github/workflows/ci.yml` without `workflow` scope
```

## Enabling it

```bash
mkdir -p .github/workflows
git mv ci/ci.yml .github/workflows/ci.yml
git rm ci/README.md
git commit -m "ci: enable GitHub Actions workflow"
git push
```

## Note on the device policy

Models run on GPU (`app/services/device.py`). GitHub runners have none, so the
workflow sets `ALLOW_CPU=1`, the one sanctioned escape, which `tests/conftest.py`
also sets locally. Training, evaluation and serving all still require CUDA.
