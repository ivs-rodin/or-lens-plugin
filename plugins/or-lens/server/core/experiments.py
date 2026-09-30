"""Validated solver wrapper and deterministic run comparison rules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import highspy

from server.core.errors import ModelError
from server.core.model_loader import LoadedModel
from server.core.solver import solve_model
from server.core.validation import unavailable_validation, validate_primal
from server.schemas.experiments import (
    MAX_REPEATS,
    MetricDeltas,
    RepeatSummary,
    RunComparison,
    RunRecord,
    SimplexStrategy,
    SolverOptions,
    ValidatedSolveParameters,
    ValidatedSolveResult,
    ValidationStatus,
)

# Relative to max(1, |objective|): an absolute 1e-7 is below floating-point
# resolution for objectives of order 1e9 and would report noise as a change.
OBJECTIVE_TOLERANCE = 1e-7
SIMPLEX_STRATEGY_CODES: dict[SimplexStrategy, int] = {
    "choose": 0,
    "dual": 1,
    "primal": 4,
}
Verdict = Literal["candidate_retained", "baseline_retained", "inconclusive"]


def _on_off(value: bool) -> str:
    return "on" if value else "off"


def _apply_search_options(highs: highspy.Highs, options: SolverOptions) -> None:
    """Set allowlisted options; none of them changes the model or its tolerances."""
    values: tuple[tuple[str, str | float | bool | int], ...] = (
        ("presolve", options.presolve),
        ("mip_heuristic_effort", float(options.mip_heuristic_effort)),
        ("mip_detect_symmetry", bool(options.mip_detect_symmetry)),
        ("mip_allow_restart", bool(options.mip_allow_restart)),
        ("simplex_strategy", SIMPLEX_STRATEGY_CODES[options.simplex_strategy]),
    )
    for name, value in values:
        if highs.setOptionValue(name, value) == highspy.HighsStatus.kError:
            raise ModelError("solver_error", f"HiGHS could not configure {name}.")


def solve_validated(model: LoadedModel, params: ValidatedSolveParameters) -> ValidatedSolveResult:
    """Apply allowlisted options, solve, then validate exact original-model primal values."""
    _apply_search_options(model.highs, params)
    result = solve_model(model, params.solve)
    solution = model.highs.getSolution()
    validation = (
        validate_primal(model, solution.col_value, result.objective_value)
        if result.solution.available and solution.value_valid
        else unavailable_validation()
    )
    return ValidatedSolveResult(
        result=result,
        validation=validation,
        presolve=params.presolve,
        mip_heuristic_effort=params.mip_heuristic_effort,
        mip_detect_symmetry=params.mip_detect_symmetry,
        mip_allow_restart=params.mip_allow_restart,
        simplex_strategy=params.simplex_strategy,
    )


def describe_trial_options(options: SolverOptions, threads: int) -> str:
    """Summarize every allowlisted trial setting for people reading run history."""
    return " · ".join(
        (
            f"presolve {options.presolve}",
            f"threads {threads}",
            f"heuristics {options.mip_heuristic_effort:g}",
            f"symmetry {_on_off(options.mip_detect_symmetry)}",
            f"restart {_on_off(options.mip_allow_restart)}",
            f"simplex {options.simplex_strategy}",
        )
    )


def representative_index(runtimes: Sequence[float]) -> int:
    """Index of the lower-median runtime; equal runtimes keep execution order."""
    order = sorted(range(len(runtimes)), key=lambda index: (runtimes[index], index))
    return order[(len(order) - 1) // 2]


def _aggregate_status(statuses: Sequence[ValidationStatus]) -> ValidationStatus:
    if all(status == "valid" for status in statuses):
        return "valid"
    return "invalid" if "invalid" in statuses else "unavailable"


def _objective_tolerance(values: Sequence[float], relative: float = OBJECTIVE_TOLERANCE) -> float:
    return max(OBJECTIVE_TOLERANCE, relative) * max([1.0, *(abs(value) for value in values)])


def _consistent(
    reasons: Sequence[str], objectives: Sequence[float | None], mip_gap: float | None
) -> bool:
    """Equal termination reasons and objectives; a missing value only matches a missing one.

    ``mip_gap`` is the relative gap every repeat was solved to when all of them
    are MIP solves that ended optimal; seeds may then legitimately stop at
    different incumbents within that gap.
    """
    if len(set(reasons)) > 1:
        return False
    known = [value for value in objectives if value is not None]
    if not known:
        return True
    if len(known) != len(objectives):
        return False
    # An overflowing spread becomes inf and is therefore inconsistent.
    spread = max(known) - min(known)
    return spread <= _objective_tolerance(known, mip_gap or OBJECTIVE_TOLERANCE)


def _configuration(validated: ValidatedSolveResult) -> tuple[object, ...]:
    return (
        validated.result.model_hash,
        validated.result.solver_version,
        validated.result.effective_parameters.model_dump(exclude={"random_seed"}),
        validated.model_dump(include=set(SolverOptions.model_fields)),
    )


def _solver_parameters(validated: ValidatedSolveResult) -> dict[str, float | int | str]:
    params = validated.result.effective_parameters
    return {
        "time_limit_seconds": params.time_limit_seconds,
        "mip_relative_gap": params.mip_relative_gap,
        "threads": params.threads,
        "random_seed": params.random_seed,
        "presolve": validated.presolve,
        "mip_heuristic_effort": validated.mip_heuristic_effort,
        "mip_detect_symmetry": _on_off(validated.mip_detect_symmetry),
        "mip_allow_restart": _on_off(validated.mip_allow_restart),
        "simplex_strategy": validated.simplex_strategy,
    }


def make_run_record(
    model_id: str,
    parent_run_id: str | None,
    description: str,
    validated: ValidatedSolveResult | Sequence[ValidatedSolveResult],
    hypothesis: str | None = None,
    run_id: str | None = None,
) -> RunRecord:
    """Record one validated solve, or seed repeats of one configuration, as one run.

    Repeats keep the unmodified result of the lower-median-runtime repeat. The
    run is valid only if every repeat is valid, invalid if any repeat is invalid,
    and otherwise unavailable. Its report is the first repeat's report with that
    aggregated status, or the representative repeat's report when all are valid.
    """
    repeats = [validated] if isinstance(validated, ValidatedSolveResult) else list(validated)
    if not 1 <= len(repeats) <= MAX_REPEATS:
        raise ValueError("A run records one to five validated solves.")
    first = repeats[0]
    if any(_configuration(repeat) != _configuration(first) for repeat in repeats[1:]):
        raise ValueError("Repeats must share the model, solver and every non-seed option.")
    runtimes = [repeat.result.runtime_seconds for repeat in repeats]
    representative = repeats[representative_index(runtimes)]
    status = _aggregate_status([repeat.validation.status for repeat in repeats])
    reported = (
        representative
        if status == "valid"
        else next(repeat for repeat in repeats if repeat.validation.status == status)
    )
    summary = None
    if len(repeats) > 1:
        summary = RepeatSummary(
            repeats=len(repeats),
            seeds=[repeat.result.effective_parameters.random_seed for repeat in repeats],
            runtimes_seconds=runtimes,
            termination_reasons=[repeat.result.termination_reason for repeat in repeats],
            objective_values=[repeat.result.objective_value for repeat in repeats],
            validation_statuses=[repeat.validation.status for repeat in repeats],
            median_runtime_seconds=representative.result.runtime_seconds,
            min_runtime_seconds=min(runtimes),
            max_runtime_seconds=max(runtimes),
            representative_seed=representative.result.effective_parameters.random_seed,
            validation_seed=reported.result.effective_parameters.random_seed,
            consistent=_consistent(
                [repeat.result.termination_reason for repeat in repeats],
                [repeat.result.objective_value for repeat in repeats],
                first.result.effective_parameters.mip_relative_gap
                if all(
                    repeat.result.termination_reason == "optimal"
                    and (repeat.result.mip_gap is not None or repeat.result.best_bound is not None)
                    for repeat in repeats
                )
                else None,
            ),
        )
    return RunRecord(
        run_id=run_id or str(uuid4()),
        parent_run_id=parent_run_id,
        model_id=model_id,
        model_hash=first.result.model_hash,
        created_at=datetime.now(UTC),
        solver="HiGHS",
        solver_version=first.result.solver_version,
        solver_parameters=_solver_parameters(first),
        change_description=description,
        hypothesis=hypothesis,
        validation_status=status,
        result=representative.result,
        validation=reported.validation,
        repeat_summary=summary,
    )


def _delta(candidate: float | int | None, baseline: float | int | None) -> float | int | None:
    if candidate is None or baseline is None:
        return None
    try:
        delta = float(candidate) - float(baseline)
    except OverflowError:
        return None
    return delta if math.isfinite(delta) else None


def _node_delta(candidate: int | None, baseline: int | None) -> int | None:
    return None if candidate is None or baseline is None else candidate - baseline


def _repeat_count(run: RunRecord) -> int:
    return run.repeat_summary.repeats if run.repeat_summary is not None else 1


def _seed_schedule(run: RunRecord) -> tuple[object, ...]:
    if run.repeat_summary is not None:
        return tuple(run.repeat_summary.seeds)
    return (run.solver_parameters.get("random_seed"),)


def _termination_reasons(run: RunRecord) -> list[str]:
    if run.repeat_summary is not None:
        return run.repeat_summary.termination_reasons
    return [run.result.termination_reason]


def _objectives(run: RunRecord) -> list[float | None]:
    """Every seed's objective for repeated runs, else the single recorded objective."""
    if run.repeat_summary is not None:
        return list(run.repeat_summary.objective_values)
    return [run.result.objective_value]


def _runtime_range(run: RepeatSummary) -> str:
    return f"[{run.min_runtime_seconds:.6g}, {run.max_runtime_seconds:.6g}] s"


def compare_runs(
    baseline: RunRecord, candidate: RunRecord, objective_sense: Literal["minimize", "maximize"]
) -> RunComparison:
    """Conservatively retain only valid, comparable objective/gap/runtime improvements.

    Repeated runs must agree across seeds before objective, gap or runtime rules
    apply to their representative results. A runtime win then requires disjoint
    observed runtime ranges; single runs keep the single-observation caveat.
    """
    notes: list[str] = []
    if baseline.model_hash != candidate.model_hash:
        notes.append("Model hashes differ.")
    if baseline.solver_version != candidate.solver_version:
        notes.append("Solver versions differ.")
    for key in ("time_limit_seconds", "mip_relative_gap"):
        if baseline.solver_parameters.get(key) != candidate.solver_parameters.get(key):
            notes.append(f"Solver parameter {key} differs.")
    same_repeats = _repeat_count(baseline) == _repeat_count(candidate)
    if baseline.solver_parameters.get("random_seed") != candidate.solver_parameters.get(
        "random_seed"
    ) or (same_repeats and _seed_schedule(baseline) != _seed_schedule(candidate)):
        notes.append("Solver parameter random_seed differs.")
    if not same_repeats:
        notes.append("Repeat counts differ.")
    comparable = not notes
    deltas = MetricDeltas(
        objective=_delta(candidate.result.objective_value, baseline.result.objective_value),
        bound=_delta(candidate.result.best_bound, baseline.result.best_bound),
        gap=_delta(candidate.result.mip_gap, baseline.result.mip_gap),
        runtime=_delta(candidate.result.runtime_seconds, baseline.result.runtime_seconds),
        nodes=_node_delta(candidate.result.node_count, baseline.result.node_count),
    )

    def decide(verdict: Verdict, better_run_id: str | None, reason: str) -> RunComparison:
        return RunComparison(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            comparable=comparable,
            comparability_notes=notes,
            deltas=deltas,
            better_run_id=better_run_id,
            verdict=verdict,
            reason=reason,
        )

    if not comparable:
        return decide(
            "inconclusive", None, "Runs do not share model, solver version, and comparison budget."
        )
    if candidate.validation_status != "valid":
        if baseline.validation_status != "valid":
            return decide(
                "inconclusive",
                None,
                "Neither run has an independently validated feasible solution.",
            )
        return decide("baseline_retained", baseline.run_id, "Candidate validation is not valid.")
    if baseline.validation_status != "valid":
        if "infeasible" in _termination_reasons(baseline):
            notes.append("Baseline reports infeasible while candidate has a valid primal solution.")
            return decide(
                "inconclusive",
                None,
                "Contradictory feasibility statuses require explicit investigation.",
            )
        return decide(
            "candidate_retained",
            candidate.run_id,
            "Candidate is first independently validated feasible solution.",
        )
    inconsistent = [
        label
        for label, run in (("Baseline", baseline), ("Candidate", candidate))
        if run.repeat_summary is not None and not run.repeat_summary.consistent
    ]
    if inconsistent:
        notes.extend(
            f"{label} repeats disagree across seeds on termination reason or objective."
            for label in inconsistent
        )
        return decide(
            "inconclusive",
            None,
            "Repeated solves disagree across seeds; no configuration effect is established.",
        )
    # With seed repeats every seed's objective counts: an objective improvement
    # must hold for all of them, not only for the recorded median repeat.
    base_values, candidate_values = _objectives(baseline), _objectives(candidate)
    known = [value for value in base_values + candidate_values if value is not None]
    complete = len(known) == len(base_values) + len(candidate_values)
    tolerance = _objective_tolerance(known) if complete else 0.0
    base_known = [value for value in base_values if value is not None]
    candidate_known = [value for value in candidate_values if value is not None]
    improved = complete and (
        max(candidate_known) < min(base_known) - tolerance
        if objective_sense == "minimize"
        else min(candidate_known) > max(base_known) + tolerance
    )
    same_objective = complete and max(known) - min(known) <= tolerance
    lower_gap = (
        baseline.result.mip_gap is not None
        and candidate.result.mip_gap is not None
        and candidate.result.mip_gap < baseline.result.mip_gap - OBJECTIVE_TOLERANCE
    )
    if improved or (same_objective and lower_gap):
        reason = (
            (
                "Candidate improves objective on every seed."
                if len(known) > 2
                else "Candidate improves objective."
            )
            if improved
            else "Candidate matches objective with lower MIP gap."
        )
        return decide("candidate_retained", candidate.run_id, reason)
    if (
        same_objective
        and baseline.result.termination_reason == candidate.result.termination_reason == "optimal"
    ):
        base_repeats, candidate_repeats = baseline.repeat_summary, candidate.repeat_summary
        if base_repeats is None and candidate_repeats is None:
            if candidate.result.runtime_seconds < baseline.result.runtime_seconds:
                notes.append(
                    "Single-run runtime difference is observed, not a performance guarantee."
                )
                return decide(
                    "candidate_retained",
                    candidate.run_id,
                    "Candidate has equal optimal objective and lower observed runtime.",
                )
        elif (
            base_repeats is not None
            and candidate_repeats is not None
            and min(base_repeats.repeats, candidate_repeats.repeats) >= 2
        ):
            ranges = (
                f"candidate {_runtime_range(candidate_repeats)}, "
                f"baseline {_runtime_range(base_repeats)}"
            )
            if candidate_repeats.max_runtime_seconds < base_repeats.min_runtime_seconds:
                notes.append(
                    f"Observed runtime ranges across {candidate_repeats.repeats} seeds do not "
                    f"overlap ({ranges}); an observation, not a performance guarantee."
                )
                return decide(
                    "candidate_retained",
                    candidate.run_id,
                    "Candidate has equal optimal objective and a lower observed runtime range "
                    "across seeds.",
                )
            notes.append(
                f"Observed runtime ranges across {candidate_repeats.repeats} seeds overlap "
                f"({ranges})."
            )
            return decide(
                "baseline_retained",
                baseline.run_id,
                "Runtime ranges across seeds overlap; no runtime improvement is established.",
            )
    return decide(
        "baseline_retained",
        baseline.run_id,
        "Candidate did not improve retained objective or MIP gap.",
    )
