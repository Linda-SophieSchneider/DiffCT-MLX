# Releasing diffct_mlx

This repository is set up to publish the package with GitHub Actions and PyPI
Trusted Publishing.

## Package name

The import name is:

```bash
import diffct_mlx
```

The install command for users is:

```bash
pip install diffct_mlx
```

PyPI normalizes `_` and `-`, so the canonical project page may appear as
`diffct-mlx` even though `pip install diffct_mlx` works.

## Local validation

Use the repository virtual environment:

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install build twine
./.venv/bin/python -m build
./.venv/bin/python -m twine check dist/*
```

Optional install test:

```bash
./.venv/bin/python -m pip install --force-reinstall dist/*.whl
./.venv/bin/python -c "import diffct_mlx; print(diffct_mlx.__version__)"
```

## Version bump

Update the version in:

- `diffct_mlx/__about__.py`

Use a release version like `1.0.0`, `1.0.1`, `1.1.0`.
Use a development version like `1.1.0.dev0` only before release.

## GitHub Actions publishing

The publishing workflow is:

- `.github/workflows/publish.yml`

It supports:

- manual publishing to TestPyPI
- automatic publishing to PyPI on tag push `v*`

## First-time manual setup

These steps cannot be completed from this workspace. They must be done in your
GitHub and PyPI/TestPyPI accounts.

### 1. Create the package on TestPyPI

Run the workflow manually once with target `testpypi`, or upload once from your
machine with `twine`.

### 2. Configure GitHub environments

In GitHub repository settings, create these environments:

- `testpypi`
- `pypi`

Optional but sensible:

- require manual approval for `pypi`
- leave `testpypi` without approval for faster dry runs

### 3. Configure Trusted Publishers

On TestPyPI and PyPI, add a Trusted Publisher for:

- Owner/repo: `Linda-SophieSchneider/diffct_arbit`
- Workflow file: `.github/workflows/publish.yml`
- Environment: `testpypi` or `pypi`

Create one publisher entry for TestPyPI and one for PyPI.

## Release process

### Test release

1. Ensure `diffct_mlx/__about__.py` has the intended version.
2. Commit and push to `main`.
3. Run the GitHub Actions workflow `Publish Package` manually with target `testpypi`.
4. Verify installation from TestPyPI.

Example:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple diffct_mlx
```

### Production release

1. Ensure `diffct_mlx/__about__.py` contains a non-dev version such as `1.0.0`.
2. Commit and push to `main`.
3. Create and push a Git tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

4. Wait for the `Publish Package` workflow to finish.
5. Verify:

```bash
python3 -m pip install --upgrade diffct_mlx
```

## Fallback manual upload

If GitHub Actions publishing is unavailable, publish locally:

```bash
./.venv/bin/python -m build
./.venv/bin/python -m twine upload dist/*
```
