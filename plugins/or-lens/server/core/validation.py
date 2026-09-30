"""Independent primal feasibility and objective checks against original model data."""

from __future__ import annotations

import math
from typing import Literal

import highspy
import numpy as np

from server.core.model_loader import LoadedModel
from server.schemas.experiments import (
    ValidationReport,
    ValidationTolerances,
    ViolationCounts,
    ViolationExample,
)

VIOLATION_EXAMPLE_LIMIT = 20
DEFAULT_TOLERANCES = ValidationTolerances(row=1e-7, bound=1e-7, integrality=1e-7, objective=1e-7)


def unavailable_validation() -> ValidationReport:
    return ValidationReport(
        status="unavailable",
        objective_computed=None,
        objective_difference=None,
        max_row_violation=None,
        max_bound_violation=None,
        max_integrality_violation=None,
        counts=ViolationCounts(row=0, bound=0, integrality=0),
        examples=[],
        examples_truncated=False,
        example_limit=VIOLATION_EXAMPLE_LIMIT,
        tolerances=DEFAULT_TOLERANCES,
    )


def validate_primal(
    model: LoadedModel, values: list[float] | np.ndarray, reported_objective: float | None
) -> ValidationReport:
    """Check a full primal vector without trusting solver feasibility flags."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (model.lp.num_col_,) or not np.isfinite(vector).all():
        return unavailable_validation()

    with np.errstate(over="ignore", invalid="ignore"):
        row_values = np.asarray(model.matrix @ vector, dtype=np.float64).reshape(-1)
        row_lower = np.asarray(model.lp.row_lower_, dtype=np.float64)
        row_upper = np.asarray(model.lp.row_upper_, dtype=np.float64)
        col_lower = np.asarray(model.lp.col_lower_, dtype=np.float64)
        col_upper = np.asarray(model.lp.col_upper_, dtype=np.float64)
        row_violation = np.maximum(np.maximum(row_lower - row_values, row_values - row_upper), 0.0)
        bound_violation = np.maximum(np.maximum(col_lower - vector, vector - col_upper), 0.0)
    if not (
        np.isfinite(row_values).all()
        and np.isfinite(row_violation).all()
        and np.isfinite(bound_violation).all()
    ):
        return unavailable_validation()
    integrality = model.lp.integrality_ or [highspy.HighsVarType.kContinuous] * model.lp.num_col_
    integer_indices = np.asarray(
        [index for index, kind in enumerate(integrality) if kind == highspy.HighsVarType.kInteger],
        dtype=int,
    )
    integer_violation = np.zeros(model.lp.num_col_, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if len(integer_indices):
            integer_violation[integer_indices] = np.abs(
                vector[integer_indices] - np.rint(vector[integer_indices])
            )
        objective = float(
            np.dot(np.asarray(model.lp.col_cost_, dtype=np.float64), vector) + model.lp.offset_
        )
    if not np.isfinite(integer_violation).all() or not math.isfinite(objective):
        return unavailable_validation()
    with np.errstate(over="ignore", invalid="ignore"):
        objective_difference = (
            abs(objective - reported_objective)
            if reported_objective is not None and math.isfinite(reported_objective)
            else None
        )
    if objective_difference is not None and not math.isfinite(objective_difference):
        return unavailable_validation()
    examples: list[ViolationExample] = []
    total = 0
    checks: tuple[
        tuple[Literal["row", "bound", "integrality"], np.ndarray, list[str], float], ...
    ] = (
        ("row", row_violation, model.row_names, DEFAULT_TOLERANCES.row),
        ("bound", bound_violation, model.column_names, DEFAULT_TOLERANCES.bound),
        ("integrality", integer_violation, model.column_names, DEFAULT_TOLERANCES.integrality),
    )
    for kind, violations, names, tolerance in checks:
        for index in np.flatnonzero(violations > tolerance):
            total += 1
            if len(examples) < VIOLATION_EXAMPLE_LIMIT:
                examples.append(
                    ViolationExample(
                        kind=kind,
                        index=int(index),
                        name=names[index],
                        violation=float(violations[index]),
                    )
                )
    row_count = int(np.count_nonzero(row_violation > DEFAULT_TOLERANCES.row))
    bound_count = int(np.count_nonzero(bound_violation > DEFAULT_TOLERANCES.bound))
    integer_count = int(np.count_nonzero(integer_violation > DEFAULT_TOLERANCES.integrality))
    valid = not (row_count or bound_count or integer_count) and (
        objective_difference is None or objective_difference <= DEFAULT_TOLERANCES.objective
    )
    return ValidationReport(
        status="valid" if valid else "invalid",
        objective_computed=objective,
        objective_difference=objective_difference,
        max_row_violation=float(row_violation.max(initial=0.0)),
        max_bound_violation=float(bound_violation.max(initial=0.0)),
        max_integrality_violation=float(integer_violation.max(initial=0.0)),
        counts=ViolationCounts(row=row_count, bound=bound_count, integrality=integer_count),
        examples=examples,
        examples_truncated=total > VIOLATION_EXAMPLE_LIMIT,
        example_limit=VIOLATION_EXAMPLE_LIMIT,
        tolerances=DEFAULT_TOLERANCES,
    )
