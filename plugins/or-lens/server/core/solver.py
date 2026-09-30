"""Deterministic, bounded solve summaries from HiGHS."""

from __future__ import annotations

import math

import highspy

from server.core.errors import ModelError
from server.core.model_loader import LoadedModel
from server.schemas.solver import (
    EffectiveSolveParameters,
    LogSummary,
    PresolveSummary,
    SolutionSummary,
    SolutionValue,
    SolveParameters,
    SolveResult,
)

SOLUTION_VALUE_LIMIT = 50


def _finite(value: float | int) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _is_mip(model: LoadedModel) -> bool:
    return any(
        variable_type == highspy.HighsVarType.kInteger for variable_type in model.lp.integrality_
    )


def _set_option(highs: highspy.Highs, name: str, value: float | int | bool) -> None:
    if highs.setOptionValue(name, value) == highspy.HighsStatus.kError:
        raise ModelError("solver_error", "HiGHS could not configure the solve.")


def _termination_reason(status: highspy.HighsModelStatus) -> str:
    """Use HiGHS' model-status enum as the sole status source."""
    return {
        highspy.HighsModelStatus.kOptimal: "optimal",
        highspy.HighsModelStatus.kInfeasible: "infeasible",
        highspy.HighsModelStatus.kUnbounded: "unbounded",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "unbounded_or_infeasible",
        highspy.HighsModelStatus.kTimeLimit: "time_limit",
        highspy.HighsModelStatus.kIterationLimit: "iteration_limit",
        highspy.HighsModelStatus.kObjectiveBound: "objective_bound",
        highspy.HighsModelStatus.kObjectiveTarget: "objective_target",
        highspy.HighsModelStatus.kSolutionLimit: "solution_limit",
        highspy.HighsModelStatus.kInterrupt: "interrupt",
        highspy.HighsModelStatus.kMemoryLimit: "memory_limit",
        highspy.HighsModelStatus.kModelEmpty: "model_empty",
        highspy.HighsModelStatus.kLoadError: "load_error",
        highspy.HighsModelStatus.kModelError: "model_error",
        highspy.HighsModelStatus.kPresolveError: "presolve_error",
        highspy.HighsModelStatus.kSolveError: "solve_error",
        highspy.HighsModelStatus.kPostsolveError: "postsolve_error",
        highspy.HighsModelStatus.kNotset: "not_set",
        highspy.HighsModelStatus.kUnknown: "unknown",
    }.get(status, "unknown")


def _solution_summary(model: LoadedModel, highs: highspy.Highs) -> tuple[SolutionSummary, bool]:
    solution = highs.getSolution()
    info = highs.getInfo()
    available = (
        info.valid
        and solution.value_valid
        and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
        and len(solution.col_value) == model.lp.num_col_
        and all(math.isfinite(float(value)) for value in solution.col_value)
    )
    if not available:
        return (
            SolutionSummary(
                available=False,
                nonzero_variable_count=0,
                values_or_reference=[],
                values_truncated=False,
                value_limit=SOLUTION_VALUE_LIMIT,
            ),
            False,
        )
    nonzero_count = 0
    values: list[SolutionValue] = []
    for index, value in enumerate(solution.col_value):
        if value == 0.0:
            continue
        nonzero_count += 1
        if len(values) < SOLUTION_VALUE_LIMIT:
            values.append(
                SolutionValue(index=index, name=model.column_names[index], value=float(value))
            )
    return (
        SolutionSummary(
            available=True,
            nonzero_variable_count=nonzero_count,
            values_or_reference=values,
            values_truncated=nonzero_count > SOLUTION_VALUE_LIMIT,
            value_limit=SOLUTION_VALUE_LIMIT,
        ),
        True,
    )


def solve_model(model: LoadedModel, parameters: SolveParameters) -> SolveResult:
    """Solve a loaded LP/MIP and return only finite, status-supported metrics."""
    highs = model.highs
    _set_option(highs, "output_flag", False)
    _set_option(highs, "log_to_console", False)
    _set_option(highs, "time_limit", parameters.time_limit_seconds)
    _set_option(highs, "mip_rel_gap", parameters.mip_relative_gap)
    _set_option(highs, "threads", parameters.threads)
    _set_option(highs, "random_seed", parameters.random_seed)
    if highs.run() == highspy.HighsStatus.kError:
        raise ModelError("solver_error", "HiGHS could not solve the model.")

    status = highs.getModelStatus()
    info = highs.getInfo()
    solution, feasible = _solution_summary(model, highs)
    mip = _is_mip(model)
    objective = _finite(info.objective_function_value) if feasible and info.valid else None
    best_bound = _finite(info.mip_dual_bound) if mip and info.valid else None
    mip_gap = _finite(info.mip_gap) if mip and info.valid else None
    node_count = (
        int(info.mip_node_count) if mip and info.valid and info.mip_node_count >= 0 else None
    )
    runtime = _finite(highs.getRunTime())

    return SolveResult(
        model_name=model.model_name,
        model_hash=model.model_hash,
        solver_version=highs.version(),
        effective_parameters=EffectiveSolveParameters(
            time_limit_seconds=parameters.time_limit_seconds,
            mip_relative_gap=parameters.mip_relative_gap,
            threads=parameters.threads,
            random_seed=parameters.random_seed,
        ),
        status=status.name,
        termination_reason=_termination_reason(status),
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        runtime_seconds=runtime if runtime is not None else 0.0,
        node_count=node_count,
        solution=solution,
        presolve=PresolveSummary(
            rows_removed=None,
            columns_removed=None,
            note="Presolve removal counts are not collected by this adapter.",
        ),
        log_summary=LogSummary(
            raw_logs_included=False,
            message=f"HiGHS ended with {highs.modelStatusToString(status)}.",
        ),
    )
