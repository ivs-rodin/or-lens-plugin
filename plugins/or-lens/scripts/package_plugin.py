#!/usr/bin/env python3
"""Create a reviewable, allowlisted local Codex plugin archive."""

from __future__ import annotations

import argparse
import json
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXACT_FILES = (
    Path("mcp.json"),
    Path(".codex-plugin/plugin.json"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("README.md"),
    Path("docs/local-codex-plugin.md"),
)
DIRECTORIES = (Path("server"), Path("scripts"), Path("skills"))
FORBIDDEN_PARTS = {".git", ".omx", ".venv", "node_modules", "__pycache__"}


def packaged_paths(root: Path) -> list[Path]:
    paths = list(EXACT_FILES)
    for directory in DIRECTORIES:
        source_directory = root / directory
        for path in source_directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symbolic links are not allowed in a plugin archive: {path}")
            if path.is_file():
                paths.append(path.relative_to(root))
    result: list[Path] = []
    for path in paths:
        source = root / path
        if source.is_symlink():
            raise ValueError(f"symbolic links are not allowed in a plugin archive: {path}")
        if not source.is_file():
            raise FileNotFoundError(f"required plugin file is missing: {path}")
        if (
            path.parts[0] == "examples"
            or FORBIDDEN_PARTS.intersection(path.parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        result.append(path)
    return sorted(set(result))


def validate_manifest(root: Path) -> None:
    manifest = json.loads((root / ".codex-plugin/plugin.json").read_text())
    if manifest.get("name") != "or-lens" or manifest.get("mcpServers") != "./.mcp.json":
        raise ValueError("Codex manifest must name OR Lens and reference ./.mcp.json")
    mcp = json.loads((root / "mcp.json").read_text())
    server = mcp.get("mcpServers", {}).get("or-lens", {})
    if server != {
        "type": "stdio",
        "command": "./scripts/run-desktop.sh",
        "cwd": "./",
        "args": [],
    }:
        raise ValueError("portable MCP config must start the package-local desktop bootstrap")


def codex_mcp_config(portable_configuration: dict[str, object]) -> dict[str, object]:
    """Convert the portable stdio schema to Codex's companion-file schema."""
    servers = portable_configuration["mcpServers"]
    assert isinstance(servers, dict)
    server = dict(servers["or-lens"])
    server.pop("type", None)
    server["cwd"] = "."
    server["startup_timeout_sec"] = 300
    server["tool_timeout_sec"] = 900
    return {"mcpServers": {"or-lens": server}}


def plugin_files(root: Path = ROOT) -> list[tuple[Path, bytes, int]]:
    """Every file of the Codex plugin with its mode, including the generated `.mcp.json`.

    The archive and the marketplace directory are both written from this list, so
    they cannot drift apart.
    """
    validate_manifest(root)
    files: list[tuple[Path, bytes, int]] = []
    for relative in packaged_paths(root):
        contents = (root / relative).read_bytes()
        mode = 0o755 if relative == Path("scripts/run-desktop.sh") else 0o644
        files.append((relative, contents, mode))
        if relative == Path("mcp.json"):
            companion = json.dumps(codex_mcp_config(json.loads(contents)), indent=2) + "\n"
            files.append((Path(".mcp.json"), companion.encode(), 0o644))
    return files


def write_archive(output: Path) -> list[Path]:
    files = plugin_files(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, contents, mode in files:
            info = zipfile.ZipInfo(relative.as_posix())
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, contents, compress_type=zipfile.ZIP_DEFLATED)
    return [relative for relative, _, _ in files if relative != Path(".mcp.json")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/or-lens-codex-plugin.zip")
    args = parser.parse_args()
    paths = write_archive(args.output)
    print(f"wrote {args.output} ({len(paths)} files)")


if __name__ == "__main__":
    main()
