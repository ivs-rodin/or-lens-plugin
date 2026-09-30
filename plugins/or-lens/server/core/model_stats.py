"""Deterministic structural statistics for a loaded HiGHS model."""

from __future__ import annotations

from typing import Literal

import highspy
import numpy as np

from server.schemas.model import (
    CoefficientRange,
    CoefficientRanges,
    ConstraintStatistics,
    MatrixStatistics,
    ModelInspection,
    ObjectiveStatistics,
    StructureStatistics,
    VariableStatistics,
)

from .model_loader import LoadedModel


def _magnitude_range(values: np.ndarray) -> CoefficientRange:
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    nonzero = finite[finite != 0]
    return CoefficientRange(
        min_abs_nonzero=float(nonzero.min()) if nonzero.size else None,
        max_abs=float(finite.max()) if finite.size else None,
    )


def _flatten_ranges(
    matrix: CoefficientRange,
    objective: CoefficientRange,
    rhs: CoefficientRange,
    lower_bound: CoefficientRange,
    upper_bound: CoefficientRange,
) -> CoefficientRanges:
    return CoefficientRanges(
        matrix_min_abs_nonzero=matrix.min_abs_nonzero,
        matrix_max_abs=matrix.max_abs,
        objective_min_abs_nonzero=objective.min_abs_nonzero,
        objective_max_abs=objective.max_abs,
        rhs_min_abs_nonzero=rhs.min_abs_nonzero,
        rhs_max_abs=rhs.max_abs,
        lower_bound_min_abs_nonzero=lower_bound.min_abs_nonzero,
        lower_bound_max_abs=lower_bound.max_abs,
        upper_bound_min_abs_nonzero=upper_bound.min_abs_nonzero,
        upper_bound_max_abs=upper_bound.max_abs,
    )


def _variable_domains(model: LoadedModel) -> VariableStatistics:
    lp = model.lp
    lowers = np.asarray(lp.col_lower_, dtype=np.float64)
    uppers = np.asarray(lp.col_upper_, dtype=np.float64)
    integrality = list(lp.integrality_)
    if not integrality:
        integrality = [highspy.HighsVarType.kContinuous] * lp.num_col_
    integer = [kind == highspy.HighsVarType.kInteger for kind in integrality]
    binary = [
        bool(is_integer and lower >= 0 and upper <= 1)
        for is_integer, lower, upper in zip(integer, lowers, uppers, strict=True)
    ]
    infinity = 1e20
    return VariableStatistics(
        total=lp.num_col_,
        binary=sum(binary),
        integer=sum(
            is_integer and not is_binary
            for is_integer, is_binary in zip(integer, binary, strict=True)
        ),
        continuous=sum(not is_integer for is_integer in integer),
        fixed=int(np.count_nonzero(lowers == uppers)),
        free=int(np.count_nonzero((lowers <= -infinity) & (uppers >= infinity))),
    )


def _constraint_senses(model: LoadedModel) -> tuple[ConstraintStatistics, np.ndarray]:
    lp = model.lp
    lowers = np.asarray(lp.row_lower_, dtype=np.float64)
    uppers = np.asarray(lp.row_upper_, dtype=np.float64)
    infinity = 1e20
    lower_finite = lowers > -infinity
    upper_finite = uppers < infinity
    equality = lower_finite & upper_finite & (lowers == uppers)
    ranged = lower_finite & upper_finite & ~equality
    less_than = ~lower_finite & upper_finite
    greater_than = lower_finite & ~upper_finite
    free = ~lower_finite & ~upper_finite
    rhs = np.concatenate((lowers[lower_finite], uppers[upper_finite]))
    return (
        ConstraintStatistics(
            total=lp.num_row_,
            equality=int(equality.sum()),
            less_equal=int(less_than.sum()),
            greater_equal=int(greater_than.sum()),
            ranged=int(ranged.sum()),
            free=int(free.sum()),
        ),
        rhs,
    )


def inspect_model(model: LoadedModel) -> ModelInspection:
    """Return finite, sparse structural facts for an accepted model."""
    lp = model.lp
    senses, rhs = _constraint_senses(model)
    matrix = model.matrix
    row_counts = np.diff(matrix.indptr)
    column_counts = np.diff(matrix.tocsc().indptr)
    objective_sense: Literal["minimize", "maximize"] = (
        "minimize" if lp.sense_ == highspy.ObjSense.kMinimize else "maximize"
    )
    return ModelInspection(
        model_name=model.model_name,
        model_hash=model.model_hash,
        solver_version=model.highs.version(),
        objective=ObjectiveStatistics(
            sense=objective_sense,
            offset=float(lp.offset_),
        ),
        variables=_variable_domains(model),
        constraints=senses,
        matrix=MatrixStatistics(
            nonzeros=int(matrix.nnz),
            density=(float(matrix.nnz) / (lp.num_row_ * lp.num_col_))
            if lp.num_row_ and lp.num_col_
            else 0.0,
        ),
        coefficient_ranges=_flatten_ranges(
            matrix=_magnitude_range(matrix.data),
            objective=_magnitude_range(np.asarray(lp.col_cost_)),
            rhs=_magnitude_range(rhs),
            lower_bound=_magnitude_range(
                np.asarray(lp.col_lower_)[np.asarray(lp.col_lower_) > -1e20]
            ),
            upper_bound=_magnitude_range(
                np.asarray(lp.col_upper_)[np.asarray(lp.col_upper_) < 1e20]
            ),
        ),
        structure=StructureStatistics(
            empty_rows=int(np.count_nonzero(row_counts == 0)),
            empty_columns=int(np.count_nonzero(column_counts == 0)),
            singleton_rows=int(np.count_nonzero(row_counts == 1)),
        ),
    )
