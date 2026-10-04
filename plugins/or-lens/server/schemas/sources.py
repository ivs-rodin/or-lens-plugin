"""Strict, bounded schemas for STOLP infeasibility sources and their whole-model fixes."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from server.core.limits import DEFAULT_TIME_LIMIT_SECONDS
from server.schemas.base import StrictModel
from server.schemas.conflicts import ConflictSeconds

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class SourcesParameters(StrictModel):
    """Wall-clock budget for the whole STOLP search, including its whole-model fixes."""

    time_limit_seconds: ConflictSeconds = DEFAULT_TIME_LIMIT_SECONDS


class SourceChange(StrictModel):
    """Move one bound of one constraint: ``current`` becomes ``needed``."""

    constraint: str = Field(max_length=200)
    constraint_truncated: bool = False
    bound: Literal["upper", "lower", "rhs"]
    current: FiniteFloat
    needed: FiniteFloat
    shift: FiniteFloat


class SourceFix(StrictModel):
    """Change every listed bound of one constraint group together.

    ``checked_on_whole_model`` is True when STOLP verified on the whole model, with
    the other sources switched off, that this change alone removes the source.
    Otherwise it is an example computed on the conflict's own constraints and the
    whole model may need more.
    """

    group: str = Field(max_length=200)
    group_truncated: bool = False
    changes: list[SourceChange] = Field(min_length=1, max_length=3)
    checked_on_whole_model: bool


class InfeasibilitySource(StrictModel):
    """One independent cause: a combination of constraint groups at some indices."""

    groups: list[str] = Field(max_length=20)
    groups_truncated: bool = False
    where: str = Field(max_length=500)
    where_truncated: bool = False
    conflicts: Annotated[int, Field(ge=1)]
    # Alternatives, smallest relative change first; any one is enough.
    fixes: list[SourceFix] = Field(default_factory=list, max_length=6)
    fixes_truncated: bool = False
    # Set for a group combination that only carries another source over to other
    # indices; it is not a separate cause.
    carried_from: list[str] | None = Field(default=None, max_length=20)


class SourcesReport(StrictModel):
    model_hash: str
    status: Literal["infeasible", "feasible", "unknown", "unavailable"]
    method: Literal["stolp"] = "stolp"
    stolp_version: str | None = Field(default=None, max_length=50)
    message: str = Field(max_length=500)
    # Independent sources, not counting carried-over combinations.
    source_count: Annotated[int, Field(ge=0)] = 0
    sources: list[InfeasibilitySource] = Field(default_factory=list, max_length=50)
    sources_truncated: bool = False
    # Constraints the search switched off; without them the model is feasible
    # unless ``stopped`` is set.
    switched_off: list[str] = Field(default_factory=list, max_length=100)
    switched_off_total: Annotated[int, Field(ge=0)] = 0
    summary: str = Field(default="", max_length=20000)
    summary_truncated: bool = False
    # How to use the report; present when the model is infeasible.
    guidance: str = Field(default="", max_length=4000)
    stopped: Literal["max_conflicts", "max_time", "unknown", "protected"] | None = None
    solves: Annotated[int, Field(ge=0)] = 0
    solver_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    runtime_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
