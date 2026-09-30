"""Schemas for deterministic, pre-presolve model inspection."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel


class CoefficientRange(StrictModel):
    """Finite absolute-value range; minimum excludes zero."""

    min_abs_nonzero: float | None = Field(default=None, ge=0.0)
    max_abs: float | None = Field(default=None, ge=0.0)


class VariableStatistics(StrictModel):
    total: int = Field(ge=0)
    binary: int = Field(ge=0)
    integer: int = Field(ge=0)
    continuous: int = Field(ge=0)
    fixed: int = Field(ge=0)
    free: int = Field(ge=0)


class ConstraintStatistics(StrictModel):
    total: int = Field(ge=0)
    equality: int = Field(ge=0)
    less_equal: int = Field(ge=0)
    greater_equal: int = Field(ge=0)
    ranged: int = Field(ge=0)
    free: int = Field(ge=0)


class MatrixStatistics(StrictModel):
    nonzeros: int = Field(ge=0)
    density: float = Field(ge=0.0, le=1.0)


class CoefficientRanges(StrictModel):
    matrix_min_abs_nonzero: float | None = Field(default=None, ge=0.0)
    matrix_max_abs: float | None = Field(default=None, ge=0.0)
    objective_min_abs_nonzero: float | None = Field(default=None, ge=0.0)
    objective_max_abs: float | None = Field(default=None, ge=0.0)
    rhs_min_abs_nonzero: float | None = Field(default=None, ge=0.0)
    rhs_max_abs: float | None = Field(default=None, ge=0.0)
    lower_bound_min_abs_nonzero: float | None = Field(default=None, ge=0.0)
    lower_bound_max_abs: float | None = Field(default=None, ge=0.0)
    upper_bound_min_abs_nonzero: float | None = Field(default=None, ge=0.0)
    upper_bound_max_abs: float | None = Field(default=None, ge=0.0)


class StructureStatistics(StrictModel):
    empty_rows: int = Field(ge=0)
    empty_columns: int = Field(ge=0)
    singleton_rows: int = Field(ge=0)


class ObjectiveStatistics(StrictModel):
    sense: Literal["minimize", "maximize"]
    offset: float


class ModelInspection(StrictModel):
    """All facts computed deterministically from the accepted HiGHS LP."""

    model_name: str
    model_hash: str
    solver_version: str
    objective: ObjectiveStatistics
    variables: VariableStatistics
    constraints: ConstraintStatistics
    matrix: MatrixStatistics
    coefficient_ranges: CoefficientRanges
    structure: StructureStatistics
