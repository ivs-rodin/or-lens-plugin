"""Ephemeral workspace contracts shared by the browser and MCP App."""

from typing import Annotated

from pydantic import Field

from server.core.limits import DEFAULT_TIME_LIMIT_SECONDS, MAX_TIME_LIMIT_SECONDS
from server.schemas.base import StrictModel
from server.schemas.conflicts import ConflictReport
from server.schemas.diagnostics import ModelAnalysis
from server.schemas.experiments import (
    MAX_REPEATS,
    RunRecord,
    SolverOptions,
    ValidatedSolveParameters,
)
from server.schemas.model import ModelInspection
from server.schemas.solver import SolveParameters


class ModelSummary(StrictModel):
    model_id: str
    name: str
    model_hash: str
    size_bytes: int = Field(ge=0)
    created_at: str


class ModelList(StrictModel):
    models: list[ModelSummary]


class WorkspaceOverview(StrictModel):
    model_id: str
    inspection: ModelInspection
    analysis: ModelAnalysis
    latest_run: RunRecord | None = None
    latest_conflict: ConflictReport | None = None


class Example(StrictModel):
    name: str
    label: str
    description: str


class ExampleList(StrictModel):
    examples: list[Example]


class DeleteResult(StrictModel):
    deleted: bool


class RunList(StrictModel):
    runs: list[RunRecord] = Field(max_length=100)


class RunParameters(SolveParameters, SolverOptions):
    """A manual solve: comparison settings plus allowlisted HiGHS search options."""

    def validated_parameters(self) -> ValidatedSolveParameters:
        return ValidatedSolveParameters.model_validate(
            {
                **self.model_dump(include=set(SolverOptions.model_fields)),
                "solve": self.model_dump(include=set(SolveParameters.model_fields)),
            }
        )


class StartExperimentParameters(StrictModel):
    """A session's fixed comparison settings.

    With repeats k > 1, the baseline and every trial each run k validated solves
    with seeds 0..k-1 and record one run.
    """

    model_id: str = Field(min_length=1, max_length=128)
    max_runs: Annotated[int, Field(ge=1, le=10, strict=True)] = 3
    time_limit_seconds: Annotated[
        float,
        Field(ge=0, le=MAX_TIME_LIMIT_SECONDS, strict=True, allow_inf_nan=False),
    ] = DEFAULT_TIME_LIMIT_SECONDS
    repeats: Annotated[int, Field(ge=1, le=MAX_REPEATS, strict=True)] = 1


class TrialParameters(SolverOptions):
    """One hypothesis and the allowlisted options it changes."""

    hypothesis: str = Field(min_length=1, max_length=1000)
    threads: Annotated[int, Field(ge=1, le=4, strict=True)] = 1


class CompareParameters(StrictModel):
    baseline_run_id: str = Field(min_length=1, max_length=128)
    candidate_run_id: str = Field(min_length=1, max_length=128)
