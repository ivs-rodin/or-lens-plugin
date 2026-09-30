"""Strict, bounded schemas for native HiGHS LP and MIP LP-relaxation conflict reports."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from server.core.limits import DEFAULT_TIME_LIMIT_SECONDS, MAX_TIME_LIMIT_SECONDS
from server.schemas.base import StrictModel

ConflictSeconds = Annotated[
    float,
    Field(
        default=DEFAULT_TIME_LIMIT_SECONDS,
        strict=True,
        ge=0,
        le=MAX_TIME_LIMIT_SECONDS,
        allow_inf_nan=False,
    ),
]


class ConflictParameters(StrictModel):
    """Budget for an LP conflict calculation and its independent verification."""

    time_limit_seconds: ConflictSeconds = DEFAULT_TIME_LIMIT_SECONDS


class ConflictMember(StrictModel):
    index: Annotated[int, Field(ge=0)]
    name: str = Field(max_length=200)
    name_truncated: bool = False
    bound: Literal["lower", "upper", "both"]
    lower: Annotated[float, Field(allow_inf_nan=False)] | None
    upper: Annotated[float, Field(allow_inf_nan=False)] | None


class ConflictGroup(StrictModel):
    """Optional, conservative grouping derived only from an explicit name prefix."""

    prefix: str = Field(min_length=1, max_length=200)
    prefix_truncated: bool = False
    count: Annotated[int, Field(ge=1)]


class ConflictEdge(StrictModel):
    row_index: Annotated[int, Field(ge=0)]
    column_index: Annotated[int, Field(ge=0)]


class ConflictColumn(StrictModel):
    """Name of a column referenced by a returned conflict edge."""

    index: Annotated[int, Field(ge=0)]
    name: str = Field(max_length=200)
    name_truncated: bool = False


class ConflictMetadata(StrictModel):
    verified: bool
    irreducible: bool
    total_rows: Annotated[int, Field(ge=0)]
    total_bounds: Annotated[int, Field(ge=0)]
    rows_truncated: bool
    bounds_truncated: bool
    edges_truncated: bool
    iis_status: int
    iis_valid: bool
    validation_status: str
    # True when a MIP was analyzed through its LP relaxation (every column continuous).
    # IIS membership, irreducibility and verification then refer to that relaxation.
    relaxation: bool = False


class ConflictReport(StrictModel):
    model_hash: str
    status: Literal["conflict", "feasible", "unsupported", "time_limit", "unavailable"]
    method: str
    message: str
    infeasible_rows: list[ConflictMember] = Field(default_factory=list, max_length=100)
    infeasible_bounds: list[ConflictMember] = Field(default_factory=list, max_length=100)
    constraint_groups: list[ConflictGroup] = Field(default_factory=list, max_length=20)
    variable_groups: list[ConflictGroup] = Field(default_factory=list, max_length=20)
    edges: list[ConflictEdge] = Field(default_factory=list, max_length=300)
    # Distinct columns of the returned edges, in first-appearance order.
    columns: list[ConflictColumn] = Field(default_factory=list, max_length=300)
    conflict_metadata: ConflictMetadata
    runtime_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
