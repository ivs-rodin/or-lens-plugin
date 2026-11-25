# Local Codex plugin

The plugin runs OR Lens and HiGHS on the user's computer. Model files, the
matrix and computations are not sent to a VPS: Codex talks to a local MCP
process over standard input/output. Text summaries and results that Codex uses
in its answer become part of the chat context.

## Use in Codex

After installing, enable OR Lens and open a **new chat**: chats that are already
open may keep the previous set of MCP tools. Give the path to the model, for
example:

> Use OR Lens. Open /full/path/model.lp, show the panel on the right
> and find out why the model is hard to solve. Start with an analysis
> and a 30-second run.

The server and the panel start automatically. Select a row or a block in Matrix,
then ask about that selection in the chat — Codex receives it through
`get_optimization_panel_state`. Computations from the chat update Latest solve
and the run history in the open panel. The session is temporary: after the
plugin restarts, the model has to be opened again; the source file stays where
it is.

To distribute the plugin, use **Share** on the installed plugin's card in
Codex. The archive below contains the plugin runtime; it is not a separate
desktop installer.

## Build

From the root of the source repository, create a portable archive:

```sh
python3 scripts/package_plugin.py
```

It creates `dist/or-lens-codex-plugin.zip`. The archive deliberately contains
only the manifests, the server, the built UI, the bundled safe examples, the
Codex skills, the lockfile and the bootstrap script. The root `examples/`, the
user's `.mcp.json`, `.git`, `.omx`, virtual environments and user models do not
go into the archive.

The source repository contains portable `plugin.json` and `mcp.json`. The built
archive is meant specifically for Codex: it keeps `.codex-plugin/plugin.json`,
`mcp.json` for the bootstrap and the legacy `.mcp.json` for Codex generated from
it. The portable `mcp.json` uses the official stdio format `type: "stdio"` and
`cwd: "./"`; the `.mcp.json` for Codex adds Codex's startup timeout (300 seconds)
and operation timeout (900 seconds). The root `plugin.json` is left out of the
archive so that the portable manifest does not disable these Codex-specific
timeout overrides. The configuration does not substitute an absolute path to the
plugin: the supported local format runs `./scripts/run-desktop.sh` with
`cwd: "."`, that is, from the root of the installed plugin. Do not replace this
value with a developer's path.

## GitHub marketplace

A separate public marketplace repository contains only the plugin and
`.agents/plugins/marketplace.json`. It is generated from this repository into a
sibling folder; `.git` and the license there are left untouched:

```sh
python3 scripts/export_marketplace.py ../or-lens-plugin
```

The plugin files in the marketplace match the archive from `package_plugin.py`
byte for byte. The marketplace README is `marketplace/README.md`. Users add the
marketplace and install the plugin like this:

```sh
codex plugin marketplace add ivs-rodin/or-lens-plugin
codex plugin add or-lens@or-lens
```

To update after a new export and push: `codex plugin marketplace upgrade or-lens`.

The same repository is also a Claude Code marketplace
(`.claude-plugin/marketplace.json`):

```sh
claude plugin marketplace add ivs-rodin/or-lens-plugin
claude plugin install or-lens@or-lens
```

Claude Code first reads the plugin's `.mcp.json` (the Codex format), then
`mcpServers` from `.claude-plugin/plugin.json`; the `or-lens` server from there
replaces the Codex entry and runs `${CLAUDE_PLUGIN_ROOT}/scripts/run-desktop.sh`
regardless of the working folder. The icon and logo for the Codex plugin list
are in `assets/` (generated from `ui/brand/`), the banner and the README preview
in `marketplace/assets/`. Before publishing, run `claude plugin validate` on the
plugin folder and on the marketplace root.
OpenAI's public plugin directory requires MCP over public HTTPS, so the local
stdio plugin is distributed through such a marketplace.

## First launch

On the first connection Codex runs `scripts/run-desktop.sh`. The script:

1. Checks for macOS/Linux and `arm64`/`aarch64`/`x86_64`.
2. Copies the allowlisted runtime to `$XDG_CACHE_HOME/or-lens` or
   `$HOME/.cache/or-lens`, keyed by the SHA-256 of the code and `uv.lock`.
3. Uses the installed `uv`; if there is none, downloads the official installer
   over HTTPS only, from `https://astral.sh/uv/install.sh`, into this per-user
   cache, then deletes the installer.
4. Runs `uv sync --frozen --no-dev --python 3.12` in the runtime copy and
   starts `or-lens desktop`.

The source folder of the installed plugin stays read-only. Bootstrap output goes
to stderr; stdout is reserved for the MCP protocol. The first start needs the
network for `uv` and the lockfile dependencies and can take a few minutes. The
configuration allows it 300 seconds; if the network is slower, run the bootstrap
from a terminal first and then reconnect. Later starts use the ready runtime.

`desktop` creates a local loopback UI on an arbitrary free port and serves MCP
over stdio at the same time. For resource-intensive operations Codex waits up
to 900 seconds, while individual OR Lens solver limits stay capped at five
minutes.

The bootstrap script explicitly does not support Windows yet. On an error it
exits with a clear message in the Codex logs, without a partial installation in
the plugin folder.

## Package check

```sh
python3 scripts/package_plugin.py --output /tmp/or-lens.zip
uv run pytest -q tests/test_plugin_package.py
```

After installing, start a new Codex chat and ask it to open a small `.lp`
file. Check that OR Lens appears on the right, then ask it to explain the
infeasibility and open the matrix. Check large models once the runtime is
ready, so that the time of the first dependency download is not mixed with the
analysis time.
