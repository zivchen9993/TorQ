#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def project_field(text: str, field: str) -> str:
    section_match = re.search(r"(?ms)^\[project\]\n(.*?)(?:^\[|\Z)", text)
    if not section_match:
        raise RuntimeError("Could not find [project] section in pyproject.toml")

    section = section_match.group(1)
    field_match = re.search(rf'^\s*{re.escape(field)}\s*=\s*"([^"]+)"\s*$', section, re.MULTILINE)
    if not field_match:
        raise RuntimeError(f'Could not find project field "{field}" in pyproject.toml')

    return field_match.group(1)


def tool_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def fail(message: str) -> None:
    print(f"ERROR: {message}")


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def ok(message: str) -> None:
    print(f"OK: {message}")


def main() -> int:
    text = PYPROJECT.read_text(encoding="utf-8")
    package_name = project_field(text, "name")
    version = project_field(text, "version")

    errors = 0

    branch = run("git", "branch", "--show-current")
    current_branch = branch.stdout.strip()
    if current_branch != "main":
        fail(f'Current branch is "{current_branch}", expected "main".')
        errors += 1
    else:
        ok("Current branch is main.")

    status = run("git", "status", "--short")
    if status.stdout.strip():
        fail("Worktree is dirty. Commit, stash, or discard changes before publishing.")
        errors += 1
    else:
        ok("Worktree is clean.")

    head = run("git", "rev-parse", "HEAD")
    origin_main = run("git", "rev-parse", "origin/main")
    if head.returncode != 0 or origin_main.returncode != 0:
        warn("Could not compare HEAD to origin/main. Run `git fetch origin` and verify before publishing.")
    elif head.stdout.strip() != origin_main.stdout.strip():
        fail("HEAD does not match origin/main. Publish only after main is pushed.")
        errors += 1
    else:
        ok("HEAD matches origin/main.")

    for tool_name in ("build", "twine"):
        if not tool_installed(tool_name):
            fail(f'Python module "{tool_name}" is not installed in the current environment.')
            errors += 1
        else:
            ok(f'Python module "{tool_name}" is installed.')

    if not tool_installed("torch"):
        warn('Python module "torch" is not installed. Runtime tests will be skipped.')
    else:
        ok('Python module "torch" is installed.')

    if not tool_installed("pennylane"):
        warn('Python module "pennylane" is not installed. PennyLane parity tests will be skipped.')
    else:
        ok('Python module "pennylane" is installed.')

    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            ok(f'Package "{package_name}" does not exist on PyPI yet.')
        else:
            fail(f"Could not query PyPI for {package_name}: HTTP {exc.code}.")
            errors += 1
            data = None
    except OSError as exc:
        fail(f"Could not query PyPI for {package_name}: {exc}.")
        errors += 1
        data = None

    if data is not None:
        latest = data.get("info", {}).get("version")
        releases = data.get("releases", {})
        if latest:
            ok(f'PyPI latest version is "{latest}". Local version is "{version}".')
        if version in releases and releases[version]:
            fail(f'Version "{version}" already exists on PyPI.')
            errors += 1
        else:
            ok(f'PyPI does not have version "{version}" yet.')

    if errors:
        print(f"\nPreflight failed with {errors} error(s).")
        return 1

    print(f'\nPreflight passed for {package_name} {version}.')
    print("Next steps: run pytest, build the artifacts, run twine check, then upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
