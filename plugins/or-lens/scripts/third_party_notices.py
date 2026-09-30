#!/usr/bin/env python3
"""Write THIRD_PARTY_NOTICES.md for the npm packages bundled into the workbench UI.

Runtime (non-dev) packages come from package-lock.json; each notice reproduces
the package's own license file from node_modules. Python dependencies are not
redistributed: uv installs them from PyPI on the user's computer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "THIRD_PARTY_NOTICES.md"
LICENSE_FILE = re.compile(r"^(licen[cs]e|copying)(\.[a-z]+)?$", re.IGNORECASE)


def runtime_packages() -> list[tuple[str, str, str, Path]]:
    lock = json.loads((ROOT / "package-lock.json").read_text())
    packages = []
    for path, info in lock["packages"].items():
        if not path or info.get("dev"):
            continue
        folder = ROOT / path
        name = path.removeprefix("node_modules/")
        packages.append((name, info["version"], info.get("license", "UNKNOWN"), folder))
    return sorted(packages)


def notices() -> str:
    parts = [
        "# Third-party notices\n",
        "The OR Lens workbench UI (`server/static/index.html`) is built with the npm",
        "packages below. Their license texts follow as published in each package.",
        "Regenerate with `python3 scripts/third_party_notices.py` after `npm ci`.\n",
    ]
    for name, version, license_id, folder in runtime_packages():
        files = sorted(p for p in folder.iterdir() if LICENSE_FILE.match(p.name))
        if not files:
            raise FileNotFoundError(f"no license file for {name}")
        text = files[0].read_text(encoding="utf-8").strip()
        parts.append(f"## {name} {version} ({license_id})\n\n```text\n{text}\n```\n")
    return "\n".join(parts)


if __name__ == "__main__":
    OUTPUT.write_text(notices())
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
