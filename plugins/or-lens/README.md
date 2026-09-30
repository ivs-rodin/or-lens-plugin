# OR Lens

OR Lens inspects, diagnoses, solves, and compares linear optimization models
deterministically. It uses HiGHS for LP/MIP parsing and solving; a language model
never establishes mathematical facts.

The local workbench is available in a normal browser and as an MCP App. It shows
real inspection data, bounded sparse matrix windows, name-based row and column
families, verified conflicts (LPs, and MIPs through their LP relaxation), validated
solver runs, and controlled parameter experiments. Model state is temporary. The
interface follows the operating-system theme in a browser and the host theme
inside an MCP App; brand assets live in `ui/brand/`.

## Use in local Codex

The Codex plugin runs OR Lens and HiGHS on the user's computer. Ask about a local
`.lp` or `.mps` file; Codex opens a bounded snapshot through
`open_local_optimization_model`, receives computed diagnostics, and opens the
returned workbench URL in its right browser panel. Follow-up questions reuse the
same model ID. `get_optimization_panel_state` returns the current overview and
the matrix region selected in the panel; solver runs made from chat also appear
in the workbench.

The plugin starts `or-lens desktop` automatically. It combines stdio MCP and a
loopback HTTP workbench on an allocated free port with one shared temporary
workspace. It does not need a VPS, DNS, upload-storage allowlist or separate AI
API key. Local file-opening tools are available only over desktop stdio, never
through the HTTP MCP endpoint. Native computation is local; tool results sent
to Codex still become part of the AI conversation.

See [local plugin installation and packaging](docs/local-codex-plugin.md).
The existing browser/CLI modes below remain available for development and use
without Codex.

## Run and test locally

Python 3.12–3.14 is supported. Install the Python dependencies once:

```sh
uv sync --extra dev
```

Start the HTTP transport:

```sh
uv run or-lens serve --transport streamable-http
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). If an older server is
already running on port 8000, stop it with `Ctrl+C` and start this command again;
it must be restarted to pick up code or UI changes. The terminal keeps running
while the workbench is open.

The built UI is included; Node is only needed when changing the frontend:

```sh
npm ci
npm run typecheck:ui
npm run build:ui
```

`/mcp` is an MCP protocol endpoint, not a browser page. A `404` at `/` or `/docs`
means the server was started before the workbench routes were added, or a different
process owns port 8000. Restart it as above. [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
contains the short interactive test guide.

The browser test flow is:

1. Load **Small LP** in the workbench. Its solve has objective value `7`.
2. Open **Matrix**, select a row, column, or block, and inspect its bounded
   context. In an MCP App the selection is made available to the host; a regular
   browser offers **Copy context**.
3. Open **Families** to see rows and columns grouped by name prefix, with the
   nonzeros of every family block; a contiguous family opens in the matrix.
4. Load **Infeasible LP**, then open **Conflicts**. The report is a native,
   independently verified conflict when HiGHS can produce one. In this small
   network the customers need 75 units while the plants make 70, so the conflict
   names seven constraints: both capacities, both warehouse balances and all
   three demands.
5. Open **Experiments**, start a session, and enter one solver-configuration
   hypothesis per trial. Compare the recorded run with its parent.

The command-line equivalents remain useful for local files:

```sh
uv run or-lens inspect tests/fixtures/tiny.lp
uv run or-lens analyze tests/fixtures/possible_big_m.lp
uv run or-lens solve tests/fixtures/tiny.lp --time-limit-seconds 30
uv run pytest
uv run ruff check .
uv run mypy server
```

## What the workbench does

The service accepts `.lp` and `.mps` files and retains temporary snapshots under
random model IDs. It displays model dimensions, domains, coefficient ranges,
structural/numerical diagnostics, and solver results. A possible Big-M result is
a heuristic candidate, never a proof.

The matrix viewer uses half-open, zero-based row and column windows. Small
windows return at most 2,048 exact sparse entries; larger windows return at most
1,024 aggregate tiles. A selected row, column, or block returns compact facts
and capped coefficient samples rather than the whole model.

The family view groups rows and columns by name prefix: the text before the
first `[` or `(`, otherwise the name without a trailing index such as `_12`. The
grouping is a heuristic and is labeled as one; every count, sense, domain, range
and family-to-family block inside it is exact. At most 200 families per axis and
2,000 blocks are returned (30 families per axis by default through MCP).

The conflict explorer asks the native HiGHS IIS implementation for a conflict and
independently re-solves the represented subsystem. For a MIP it first checks the
LP relaxation: if the relaxation is infeasible, its verified IIS also certifies
that the MIP is infeasible (irreducible for the relaxation, not necessarily for
the integer model). If the relaxation is feasible, a MIP feasibility solve
decides between feasible and an integer-only infeasibility, which is reported as
unsupported because a native LP IIS cannot isolate it. Reports make feasible,
time-limited, unavailable and unsupported cases explicit. It does not invent
conflict membership. The latest report is kept with the model snapshot and
returned in the overview, so a conflict found from chat already appears in the
Conflicts tab; **Find conflict** runs the analysis again.

Every workbench solve creates a run record with the model hash, solver version,
effective options, metrics, a solution validation report, and an optional parent
run. Validation checks the original matrix rows, variable bounds, integrality,
and objective independently of the solver's reported status. Comparisons expose
whether runs are comparable and preserve the single-run timing caveat; they are
not benchmark claims.

An experiment begins by solving its baseline. `max_runs` includes that baseline,
so a budget of `3` allows the baseline plus two trials. Each trial must state one
hypothesis and can change only allowlisted search options: `presolve` (`choose`,
`on`, `off`), `threads` (`1`–`4`), `mip_heuristic_effort` (`0`–`1`),
`mip_detect_symmetry`, `mip_allow_restart` and `simplex_strategy` (`choose`,
`dual`, `primal`). Defaults equal the HiGHS 1.15.1 defaults. The parallel dual
simplex variants are excluded because HiGHS 1.15.1 fails on a one-row LP with
them. The model bytes, model hash, time budget, MIP gap and seeds remain fixed.

A session may use `repeats` from 1 to 5. Each run then performs one validated
solve per seed `0..k-1` and records one run with a `repeat_summary`, so it takes
about `k ×` the time limit. The run's `result` is the real solve with the
lower-median runtime. Seeds agree when objectives match to a relative 1e-7, or
within the MIP gap when every seed of a MIP ended optimal. An objective win must
hold for every seed, and a runtime win needs non-overlapping runtime ranges;
seed disagreement makes a comparison inconclusive. A trial is independently
validated and compared to the current retained run before it is kept or
rejected. Failed or interrupted launched trials consume an attempt, and a failed
repeat fails its trial. Cancel a session with the UI or
`cancel_optimization_experiment`.

This is controlled solver tuning, not formulation rewriting: OR Lens makes no
claim that a changed model would be equivalent to the original one.

## MCP

Use standard I/O for a local MCP client:

```sh
uv run or-lens serve
```

For HTTP clients, start the server as above and connect to
`http://127.0.0.1:8000/mcp`. The core file tools are
`inspect_optimization_model`, `analyze_optimization_model`, and
`solve_optimization_model`. The workbench tools are:

- `open_optimization_model` and `get_optimization_overview`
- `view_optimization_matrix`, `select_optimization_region` and
  `get_optimization_families`
- `run_optimization_model`, `explain_optimization_conflict`, and
  `list_optimization_runs`
- `compare_optimization_runs`
- `start_optimization_experiment`, `get_optimization_experiment`,
  `run_optimization_trial`, and `cancel_optimization_experiment`

`open_optimization_model` associates the bundled MCP App resource
`ui://or-lens/v1/index.html` with an uploaded model. The browser UI is served at
`/`; it uses the same typed service API. MCP tools remain usable without the UI.

### Uploaded files and local HTTP security

The file tools accept the current OpenAI file object:

```json
{
  "file": {
    "download_url": "https://trusted-upload-host.example/signed-file",
    "file_id": "uploaded-file-id",
    "file_name": "model.mps"
  }
}
```

`download_url` and `file_id` are required. `file_name` and `mime_type` are
optional strings; pass `filename` when a host omits the filename. The tools
advertise `_meta["openai/fileParams"] = ["file"]`.

Before using remote uploads, set `OR_LENS_DOWNLOAD_HOSTS` to comma-separated,
exact trusted upload-storage hostnames from the host integration. There is no
wildcard or arbitrary-URL download path. Downloads use HTTPS and TLS validation,
pin a validated public IP, reject redirects and proxies, and stream at most
1 GiB within 300 seconds. Signed URLs are neither logged nor retained.

The local server defaults to loopback and has no authentication. For a ChatGPT
development tunnel, set `OR_LENS_HTTP_HOSTS` to its exact hostname (and port,
when nonstandard), retain host-header protection, and connect the tunnel's HTTPS
`/mcp` URL. Do not expose this server as an unrestricted public solving service.
Run untrusted workloads inside an OS or container memory limit.

## Limits and lifecycle

- One model file is at most 1 GiB (1,073,741,824 bytes). Uploads and parser snapshots
  stream to disk rather than buffering entire files in Python memory.
  Parsed models are limited to 1 million rows,
  1 million columns, and 10 million nonzeros; matrices remain sparse.
- A temporary server workspace holds at most 8 models / 4 GiB, 100 run records,
  and 20 experiment sessions. Delete a model explicitly to remove its associated
  runs and sessions. All remaining state disappears when the server stops.
- Two model operations run at once across browser and MCP requests. Upload,
  parsing and analysis each have a 300-second deadline. These are separate stage
  budgets, not a five-minute limit for the entire upload-to-result request.
  Solves and conflict extraction get their configured limit plus five seconds
  for worker completion.
- Solves, conflict extraction and experiment runs default to 300 seconds
  (five minutes), also their maximum; a run with seed repeats makes one such
  solve per seed. Solver defaults remain MIP relative gap `0.0001`, one thread
  and seed zero; up to four threads are supported.
- Output numbers are finite JSON values or `null`. Error responses use stable
  codes and safe messages; no signed URL is returned in an error.

## Validation

Run the local checks after changes:

```sh
uv run pytest
uv run ruff check .
uv run mypy server
npm run typecheck:ui
npm run build:ui
```

The 280-test suite covers parser/diagnostic/solver contracts, sparse matrix and
selection bounds, name-based families, conflict verification (LP and MIP
relaxation) and retained reports, solution validation, allowlisted options and seed repeats, run
lifecycle and budgets, large-file streaming, linear-time name extraction, upload
cancellation, time limits, icons and HTTP/MCP contracts. Frontend
typechecking/build and browser flows (upload, solve, selection, families,
conflicts, repeated experiments, light/dark themes, 375 px layout) were also
checked.
Live ChatGPT registration,
attachment acceptance, tunnel behavior, and its embedded bridge require a real
account integration and are not established by local checks.

## References

- [OpenAI ChatGPT UI guide](https://developers.openai.com/plugins/build/chatgpt-ui)
- [OpenAI plugin reference](https://developers.openai.com/plugins/reference)
- [MCP Apps API](https://apps.extensions.modelcontextprotocol.io/api/)
- [HiGHS Python API](https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/)

See [PLAN.md](PLAN.md) for accepted M4–M13 scope, [DESIGN.md](DESIGN.md) for
the brand and interface rules, and [docs/ui-contract.md](docs/ui-contract.md)
for the browser/MCP App contract.
