"""Typed, finite domain contracts for validated solves and experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from server.core.limits import MAX_TIME_LIMIT_SECONDS
from server.schemas.base import StrictModel
from server.schemas.solver import MAX_RANDOM_SEED, SolveParameters, SolveResult

MAX_REPEATS = 5
DEFAULT_MIP_HEURISTIC_EFFORT = 0.05

PresolveChoice = Literal["choose", "on", "off"]
SimplexStrategy = Literal["choose", "dual", "primal"]
ValidationStatus = Literal["valid", "invalid", "unavailable"]
HeuristicEffort = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
StrictBool = Annotated[bool, Field(strict=True)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Seed = Annotated[int, Field(ge=0, le=MAX_RANDOM_SEED)]


class SolverOptions(StrictModel):
    """Allowlisted HiGHS options that change solver search, never the model.

    Defaults are the HiGHS 1.15.1 defaults, so omitted fields change nothing.
    Tolerances, MIP gap, time limit and seed are deliberately excluded.
    simplex_strategy maps to HiGHS values choose 0, dual 1 and primal 4. The
    parallel dual variants (2 PAMI, 3 SIP) are excluded: in HiGHS 1.15.1 they end
    without a solution, or run to the time limit, on a one-row LP without presolve.
    """

    presolve: PresolveChoice = "choose"
    mip_heuristic_effort: HeuristicEffort = DEFAULT_MIP_HEURISTIC_EFFORT
    mip_detect_symmetry: StrictBool = True
    mip_allow_restart: StrictBool = True
    simplex_strategy: SimplexStrategy = "dual"


class ValidatedSolveParameters(SolverOptions):
    solve: SolveParameters = Field(default_factory=SolveParameters)


class ViolationCounts(StrictModel):
    row: Annotated[int, Field(ge=0)]
    bound: Annotated[int, Field(ge=0)]
    integrality: Annotated[int, Field(ge=0)]


class ViolationExample(StrictModel):
    kind: Literal["row", "bound", "integrality"]
    index: Annotated[int, Field(ge=0)]
    name: str
    violation: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ValidationTolerances(StrictModel):
    row: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bound: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    integrality: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    objective: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ValidationReport(StrictModel):
    status: ValidationStatus
    objective_computed: Annotated[float, Field(allow_inf_nan=False)] | None
    objective_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    max_row_violation: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    max_bound_violation: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    max_integrality_violation: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    counts: ViolationCounts
    examples: list[ViolationExample]
    examples_truncated: bool
    example_limit: Annotated[int, Field(ge=1)]
    tolerances: ValidationTolerances


class ValidatedSolveResult(SolverOptions):
    """One validated solve and the allowlisted options applied before it."""

    result: SolveResult
    validation: ValidationReport


class RepeatSummary(StrictModel):
    """Per-seed facts of one run recorded from k >= 2 independent validated solves.

    Lists share execution order (ascending seeds). The run's `result` is the
    unmodified result of the repeat with the lower-median runtime
    (`representative_seed`); for even k that is the lower middle runtime, never
    an average. `validation_seed` names the repeat whose ValidationReport the run
    records. `consistent` means equal termination reasons and objectives that
    agree within 1e-7 relative to max(1, |objective|), or within the session's MIP
    relative gap when every repeat of a MIP ended optimal, which is all the solver
    guarantees across seeds (a missing objective only matches a missing one).
    """

    repeats: Annotated[int, Field(ge=2, le=MAX_REPEATS)]
    seeds: list[Seed] = Field(max_length=MAX_REPEATS)
    runtimes_seconds: list[Seconds] = Field(max_length=MAX_REPEATS)
    termination_reasons: list[str] = Field(max_length=MAX_REPEATS)
    objective_values: list[Annotated[float, Field(allow_inf_nan=False)] | None] = Field(
        max_length=MAX_REPEATS
    )
    validation_statuses: list[ValidationStatus] = Field(max_length=MAX_REPEATS)
    median_runtime_seconds: Seconds
    min_runtime_seconds: Seconds
    max_runtime_seconds: Seconds
    representative_seed: Seed
    validation_seed: Seed
    consistent: bool

    @model_validator(mode="after")
    def _describes_each_repeat_once(self) -> Self:
        per_repeat: tuple[list, ...] = (
            self.seeds,
            self.runtimes_seconds,
            self.termination_reasons,
            self.objective_values,
            self.validation_statuses,
        )
        if any(len(values) != self.repeats for values in per_repeat):
            raise ValueError("Repeat lists must contain exactly one entry per repeat.")
        if len(set(self.seeds)) != self.repeats:
            raise ValueError("Repeat seeds must be distinct.")
        if not {self.representative_seed, self.validation_seed} <= set(self.seeds):
            raise ValueError("Recorded repeats must be among the repeat seeds.")
        runtimes = sorted(self.runtimes_seconds)
        if (
            self.min_runtime_seconds != runtimes[0]
            or self.max_runtime_seconds != runtimes[-1]
            or self.median_runtime_seconds != runtimes[(self.repeats - 1) // 2]
            or self.runtimes_seconds[self.seeds.index(self.representative_seed)]
            != self.median_runtime_seconds
        ):
            raise ValueError("Runtime statistics must describe the repeat runtimes.")
        return self


class RunRecord(StrictModel):
    """A recorded validated run.

    solver_parameters holds the options applied to every repeat: numbers for
    time_limit_seconds, mip_relative_gap, threads, random_seed (the first
    repeat's seed) and mip_heuristic_effort; strings for presolve and
    simplex_strategy; "on"/"off" for mip_detect_symmetry and mip_allow_restart.
    repeat_summary is null for a single solve.
    """

    run_id: str
    parent_run_id: str | None
    model_id: str
    model_hash: str
    created_at: datetime
    solver: str
    solver_version: str
    solver_parameters: dict[str, float | int | str]
    change_description: str
    hypothesis: str | None
    validation_status: ValidationStatus
    result: SolveResult
    validation: ValidationReport
    repeat_summary: RepeatSummary | None = None


class MetricDeltas(StrictModel):
    objective: Annotated[float, Field(allow_inf_nan=False)] | None
    bound: Annotated[float, Field(allow_inf_nan=False)] | None
    gap: Annotated[float, Field(allow_inf_nan=False)] | None
    runtime: Annotated[float, Field(allow_inf_nan=False)] | None
    nodes: int | None


class RunComparison(StrictModel):
    baseline_run_id: str
    candidate_run_id: str
    comparable: bool
    comparability_notes: list[str]
    deltas: MetricDeltas
    better_run_id: str | None
    verdict: Literal["candidate_retained", "baseline_retained", "inconclusive"]
    reason: str


class ExperimentSession(StrictModel):
    experiment_id: str
    model_id: str
    max_runs: Annotated[int, Field(ge=1, le=10)]
    time_limit_seconds: Annotated[
        float, Field(ge=0, le=MAX_TIME_LIMIT_SECONDS, allow_inf_nan=False)
    ]
    repeats: Annotated[int, Field(ge=1, le=MAX_REPEATS)] = 1
    baseline_run_id: str
    best_run_id: str
    run_ids: list[str]
    status: Literal["active", "complete", "cancelled"]
    remaining_runs: Annotated[int, Field(ge=0)]
    attempt_count: Annotated[int, Field(default=0, ge=0)] = 0
    failed_attempts: Annotated[int, Field(default=0, ge=0)] = 0
    last_error: str | None = None
    created_at: datetime


class TrialResult(StrictModel):
    session: ExperimentSession
    run: RunRecord
    comparison: RunComparison
    kept: bool
