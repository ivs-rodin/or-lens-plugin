#!/usr/bin/env python3
"""Export a Git-ready Codex marketplace that contains only the OR Lens plugin.

The output directory is meant to be its own public repository. Only the paths
this exporter owns (plugin, marketplace file, README, LICENSE) are replaced;
`.git` and any other files there stay.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_plugin  # noqa: E402

ROOT = package_plugin.ROOT
PLUGIN_DIR = Path("plugins/or-lens")
MARKETPLACE_FILE = Path(".agents/plugins/marketplace.json")
README_TEMPLATE = ROOT / "marketplace/README.md"


def marketplace_config() -> dict[str, object]:
    manifest = json.loads((ROOT / ".codex-plugin/plugin.json").read_text())
    return {
        "name": "or-lens",
        "interface": {"displayName": "OR Lens"},
        "plugins": [
            {
                "name": manifest["name"],
                "source": {"source": "local", "path": f"./{PLUGIN_DIR.as_posix()}"},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
                "category": manifest["interface"]["category"],
            }
        ],
    }


def write_marketplace(output: Path) -> list[Path]:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("export the marketplace outside the source repository")
    for owned in (
        output / PLUGIN_DIR,
        output / MARKETPLACE_FILE,
        output / "README.md",
        output / "LICENSE",
    ):
        if owned.is_dir():
            shutil.rmtree(owned)
        elif owned.exists():
            owned.unlink()
    written: list[Path] = []
    for relative, contents, mode in package_plugin.plugin_files(ROOT):
        target = output / PLUGIN_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(mode)
        written.append(PLUGIN_DIR / relative)
    marketplace = output / MARKETPLACE_FILE
    marketplace.parent.mkdir(parents=True, exist_ok=True)
    marketplace.write_text(json.dumps(marketplace_config(), indent=2) + "\n")
    shutil.copyfile(README_TEMPLATE, output / "README.md")
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
    return [*written, MARKETPLACE_FILE, Path("README.md"), Path("LICENSE")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="marketplace repository directory")
    args = parser.parse_args()
    written = write_marketplace(args.output)
    print(f"wrote {len(written)} files to {args.output}")


if __name__ == "__main__":
    main()
