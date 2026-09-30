"""Strict, bounded schemas for deterministic HiGHS solve results."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from server.core.limits import DEFAULT_TIME_LIMIT_SECONDS, MAX_TIME_LIMIT_SECONDS
from server.schemas.base import StrictModel

FiniteSeconds = Annotated[
    float,
    Field(
        default=DEFAULT_TIME_LIMIT_SECONDS,
        strict=True,
        ge=0,
        le=MAX_TIME_LIMIT_SECONDS,
        allow_inf_nan=False,
    ),
]
FiniteGap = Annotated[float, Field(default=0.0001, strict=True, ge=0, le=1, allow_inf_nan=False)]
# Upper bound of HiGHS' `random_seed` option (a 32-bit HighsInt).
MAX_RANDOM_SEED = 2_147_483_647
RandomSeed = Annotated[int, Field(default=0, strict=True, ge=0, le=MAX_RANDOM_SEED)]


class SolveParameters(StrictModel):
    """Small, reproducible subset of HiGHS solve options.

    `random_seed` is passed to HiGHS' `random_seed` option. Runs with different
    seeds are reported as not comparable.
    """

    time_limit_seconds: FiniteSeconds = DEFAULT_TIME_LIMIT_SECONDS
    mip_relative_gap: FiniteGap = 0.0001
    threads: Annotated[int, Field(default=1, strict=True, ge=1, le=4)] = 1
    random_seed: RandomSeed = 0


class EffectiveSolveParameters(SolveParameters):
    """Options applied to one solve, including the seed actually passed to HiGHS."""


class SolutionValue(StrictModel):
    index: Annotated[int, Field(ge=0)]
    name: str
    value: Annotated[float, Field(allow_inf_nan=False)]


class SolutionSummary(StrictModel):
    available: bool
    nonzero_variable_count: Annotated[int, Field(ge=0)]
    values_or_reference: list[SolutionValue]
    values_truncated: bool
    value_limit: Annotated[int, Field(ge=0)]


class PresolveSummary(StrictModel):
    rows_removed: int | None
    columns_removed: int | None
    note: str | None


class LogSummary(StrictModel):
    raw_logs_included: bool
    message: str


class SolveResult(StrictModel):
    model_name: str
    model_hash: str
    solver_version: str
    effective_parameters: EffectiveSolveParameters
    status: str
    termination_reason: str
    objective_value: Annotated[float, Field(allow_inf_nan=False)] | None
    best_bound: Annotated[float, Field(allow_inf_nan=False)] | None
    mip_gap: Annotated[float, Field(allow_inf_nan=False)] | None
    runtime_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    node_count: Annotated[int, Field(ge=0)] | None
    solution: SolutionSummary
    presolve: PresolveSummary
    log_summary: LogSummary
