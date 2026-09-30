---
name: optimization-experiments
description: Run bounded, parameter-only OR Lens solver experiments through its MCP tools when a user asks to improve a retained model without changing model semantics.
---

# Controlled OR Lens experiments

Use this skill after an optimization model is open in OR Lens and the user wants
to improve solve behavior within a stated run and time budget. This skill runs
the host-side reasoning loop; OR Lens remains the source of mathematical facts.

1. Read `get_optimization_overview` and, when useful, inspect bounded regions
   with `view_optimization_matrix` or `select_optimization_region`. Treat
   diagnostics as facts only when their schema says so; possible Big-M findings
   are heuristic candidates.
2. Start one session with `start_optimization_experiment(model_id, max_runs,
   time_limit_seconds, repeats)`. Its baseline is solved immediately and counts
   toward `max_runs`. `repeats` (1–5, default 1) makes the baseline and every
   trial run that many independent validated solves with seeds `0..repeats-1`.
   Each run still records one run and consumes one attempt, but can take about
   `repeats × time_limit_seconds` of wall time. Use more than one repeat when
   the hypothesis is about runtime.
3. Before every trial, read `get_optimization_experiment`. Stop when its status
   is not `active` or `remaining_runs` is zero.
4. State one concrete hypothesis, then call `run_optimization_trial` once. Only
   these options may change; omitted options keep the HiGHS defaults:
   - `presolve`: `choose` (default), `on`, `off`
   - `threads`: `1`–`4` (default `1`)
   - `mip_heuristic_effort`: `0`–`1` (default `0.05`)
   - `mip_detect_symmetry`, `mip_allow_restart`: `true` (default) or `false`
   - `simplex_strategy`: `choose`, `dual` (default), `primal`

   They change the solver's search, never the model. Prefer one changed option
   per trial so the hypothesis stays testable. Do not alter model text, upload a
   rewritten model, or change the time budget, MIP gap, seeds, repeats, or
   tolerances during the session. The parallel dual simplex variants are not
   offered: in HiGHS 1.15.1 they can end without a solution or run to the time
   limit on some LPs.
5. Use the returned validation and comparison to explain whether the candidate
   was retained. A repeated run is `valid` only when every repeat validated.
   Its `result` is the real result of the repeat with the lower-median runtime,
   and `repeat_summary` lists every seed's runtime, status, objective, and
   validation. A runtime win across seeds requires the candidate's slowest
   repeat to be faster than the baseline's fastest one, and an objective win
   requires every candidate seed to beat every baseline seed. Overlapping ranges
   retain the baseline. Seeds agree when their objectives match to a relative
   1e-7, or within the MIP gap when every seed of a MIP ended optimal. If seeds
   disagree on status or objective, the result is `inconclusive`; when the
   baseline's seeds disagree, every comparison against it stays inconclusive,
   so say so instead of spending the budget. A single
   run's runtime difference is only an observation. Neither is a benchmark:
   report observed ranges for the stated seeds on this machine, never a general
   speedup, and do not claim a semantic-preserving formulation rewrite.

Do not launch a trial after an error merely to compensate for its consumed
attempt. Failed or interrupted launched trials consume budget; a failure in any
repeat fails the whole trial. Stop on a closed session, exhausted budget,
cancellation, or an error that leaves no meaningful allowed parameter
hypothesis. If the user asks to stop, call `cancel_optimization_experiment`
while the session is active; no further repeats are launched.

Report the retained run ID, validation status, comparison verdict, repeat count
and observed runtime range when repeats were used, remaining budget, and the
limits of the conclusion. No backend LLM key or autonomous server loop is
required.
