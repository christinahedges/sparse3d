#!/usr/bin/env python3
"""
Generate a dynamic GitHub Actions matrix of (python, scipy) pairs.

- Python versions are provided via env PYTHONS as a JSON list, e.g. '["3.9","3.10","3.11","3.12"]'
- SciPy versions are pulled from PyPI and collapsed to the latest *patch* per minor (>= 1.10).
- A pair (py, scipy) is included only if a wheel exists for that Python (cpXY) for that SciPy release.

Writes JSON to stdout in the form:
{"include": [{"python":"3.10","scipy":"1.13.1"}, ...]}
"""

import json
import os
import sys
import urllib.request
from packaging.version import Version, InvalidVersion

PYTHONS = json.loads(os.environ.get("PYTHONS", '["3.9","3.10","3.11","3.12"]'))


def load_pypi():
    with urllib.request.urlopen("https://pypi.org/pypi/scipy/json") as r:
        return json.load(r)


def cp_tag(py: str) -> str:
    maj, minr = py.split(".")
    return f"cp{maj}{minr}"


def main():
    data = load_pypi()
    releases = data["releases"]

    # pick latest patch for each minor >= 1.10, skip pre-releases/yanked
    latest_per_minor = {}
    for sver, files in releases.items():
        try:
            v = Version(sver)
        except InvalidVersion:
            continue
        if v.is_prerelease or v < Version("1.10"):
            continue
        if not files or all(f.get("yanked", False) for f in files):
            continue
        key = (v.major, v.minor)
        if key not in latest_per_minor or v > latest_per_minor[key]:
            latest_per_minor[key] = v

    matrix = {"include": []}
    for py in PYTHONS:
        tag = cp_tag(py)
        for v in sorted(latest_per_minor.values()):
            files = releases[str(v)]
            has_wheel = any(
                f["filename"].endswith(".whl") and tag in f["filename"] for f in files
            )
            if has_wheel:
                matrix["include"].append({"python": py, "scipy": str(v)})

    json.dump(matrix, sys.stdout)


if __name__ == "__main__":
    main()
