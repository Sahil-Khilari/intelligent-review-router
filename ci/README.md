# CI workflow

`ci.yml` is the GitHub Actions workflow for this project: it installs with
uv, generates the dataset, runs the test suite, and builds both Docker
images on every pull request.

## Why it is not in `.github/workflows/`

It could not be pushed from this branch. GitHub refuses a push that creates
or updates a file under `.github/workflows/` unless the credential carries
the `workflow` scope, and the token used here did not have it:

```
refusing to allow a Personal Access Token to create or update workflow
`.github/workflows/ci.yml` without `workflow` scope
```

## Enabling it

With a token that has the `workflow` scope (or from a local clone using
SSH or the GitHub UI):

```bash
mkdir -p .github/workflows
git mv ci/ci.yml .github/workflows/ci.yml
git rm ci/README.md
git commit -m "ci: enable GitHub Actions workflow"
git push
```

The workflow runs against the TF-IDF baseline, since model weights are not
committed. That is intentional: the accuracy floor in
`tests/test_sentiment.py` has to hold for whichever model is served, so a
silent downgrade fails the build.
