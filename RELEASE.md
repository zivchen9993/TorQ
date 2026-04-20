# Releasing torq-quantum

This repo publishes the `torq-quantum` package to PyPI. Release from a clean `main` only.

## Preflight

1. Make sure the version in `pyproject.toml` is the version you intend to publish.
2. Commit and push `main`.
3. Run the repo preflight:

```bash
python scripts/release_preflight.py
```

The preflight fails if:

- the current branch is not `main`
- the worktree is dirty
- `HEAD` does not match `origin/main`
- the version in `pyproject.toml` already exists on PyPI
- `build` or `twine` is missing in the current Python environment

## Release Environment

Use a fresh virtualenv for release work:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install build twine pytest torch pennylane
python -m pip install -e .
```

`torch` is required by the package itself. `pennylane` is optional at runtime, but installing it lets you run the parity tests before shipping.

## Validate Before Upload

Run the preflight again from the release environment:

```bash
python scripts/release_preflight.py
```

Run the test suite:

```bash
pytest -q -rs
```

Build the artifacts and verify the metadata:

```bash
python -m build
python -m twine check dist/*
```

## Smoke Install The Built Wheel

This checks the exact wheel you are about to upload:

```bash
python -m venv /tmp/torq-smoke
source /tmp/torq-smoke/bin/activate
python -m pip install -U pip
python -m pip install torch
python -m pip install dist/torq_quantum-<VERSION>-py3-none-any.whl
python - <<'PY'
import importlib.metadata as md
import torq

print("metadata version:", md.version("torq-quantum"))
print("import version:", torq.__version__)
PY
deactivate
```

Replace `<VERSION>` with the version from `pyproject.toml`.

## Upload

Upload to TestPyPI first if you want an extra dry run:

```bash
python -m twine upload --repository testpypi dist/*
```

Then publish to PyPI:

```bash
python -m twine upload dist/*
```

## After Publish

Verify the public package page and install from PyPI:

```bash
python -m pip install --upgrade torq-quantum
python - <<'PY'
import importlib.metadata as md
import torq

print("metadata version:", md.version("torq-quantum"))
print("import version:", torq.__version__)
PY
```

If the release is correct, tag it:

```bash
git tag -a v<VERSION> -m "Release <VERSION>"
git push origin main --tags
```
