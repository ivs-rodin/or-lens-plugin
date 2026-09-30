"""Typed MCP actions for the workbench and a budgeted ChatGPT workflow."""

from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any, Literal

from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from server.core.errors import ModelError
from server.core.limits import DEFAULT_TIME_LIMIT_SECONDS, MAX_TIME_LIMIT_SECONDS
from server.schemas.base import ErrorDetail, ErrorResult
from server.schemas.conflicts import ConflictParameters, ConflictReport
from server.schemas.experiments import (
    DEFAULT_MIP_HEURISTIC_EFFORT,
    MAX_REPEATS,
    ExperimentSession,
    RunComparison,
    RunRecord,
    SimplexStrategy,
    TrialResult,
)
from server.schemas.families import FamilyParameters, ModelFamilies
from server.schemas.files import UploadedFile
from server.schemas.matrix import (
    MatrixParameters,
    MatrixView,
    SelectionContext,
    SelectionParameters,
)
from server.schemas.solver import MAX_RANDOM_SEED
from server.schemas.workspace import (
    RunList,
    RunParameters,
    StartExperimentParameters,
    TrialParameters,
    WorkspaceOverview,
)
from server.service import WorkbenchService
from server.web import UI_URI


def workbench_tools(service: WorkbenchService) -> list[Tool]:
    tools: list[Tool] = []

    def register(
        *,
        file: bool = False,
        render: bool = False,
        read_only: bool = True,
    ) -> Callable:
        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(function)
            async def guarded(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await function(*args, **kwargs)
                except ModelError as exc:
                    raise ToolError(
                        ErrorResult(
                            error=ErrorDetail(code=exc.code, message=exc.message)
                        ).model_dump_json()
                    ) from None
                except ValidationError:
                    raise ToolError(
                        '{"error":{"code":"invalid_parameters","message":"Invalid parameters."}}'
                    ) from None
                except Exception:
                    raise ToolError(
                        '{"error":{"code":"internal_error",'
                        '"message":"Could not complete operation."}}'
                    ) from None

            meta: dict[str, Any] = {}
            if file:
                meta["openai/fileParams"] = ["file"]
            if render:
                meta["ui"] = {"resourceUri": UI_URI, "visibility": ["model", "app"]}
            tool = Tool.from_function(
                guarded,
                meta=meta,
                annotations=ToolAnnotations(
                    readOnlyHint=read_only,
                    destructiveHint=False,
                    idempotentHint=read_only,
                    openWorldHint=False,
                ),
            )
            tool.fn_metadata.arg_model.model_config["hide_input_in_errors"] = True
            tool.fn_metadata.arg_model.model_rebuild(force=True)
            tools.append(tool)
            return guarded

        return decorate

    @register(file=True, render=True, read_only=False)
    async def open_optimization_model(
        file: UploadedFile,
        filename: Annotated[str | None, Field(max_length=512)] = None,
    ) -> WorkspaceOverview:
        """Open an LP/MPS file in the interactive OR Lens workbench.

        Inspect/diagnose the file and retain a temporary model ID for matrix,
        selection, conflicts, solving and controlled experiments. Does not solve.
        """
        return await service.open_upload(file, filename)

    @register(render=True)
    async def get_optimization_overview(model_id: str) -> WorkspaceOverview:
        """Show a previously opened model's overview, diagnostics and latest run."""
        return service.overview(model_id)

    @register()
    async def view_optimization_matrix(
        model_id: str,
        row_start: int = 0,
        row_end: int | None = None,
        column_start: int = 0,
        column_end: int | None = None,
    ) -> MatrixView:
        """Return a bounded sparse matrix window, aggregating large windows into tiles.

        Indices are zero-based, end indices exclusive. Never returns all entries
        of a large model. Use select_optimization_region for concise context.
        """
        return await service.matrix(
            model_id,
            MatrixParameters(
                row_start=row_start,
                row_end=row_end,
                column_start=column_start,
                column_end=column_end,
            ),
        )

    @register()
    async def select_optimization_region(
        model_id: str,
        kind: Literal["row", "column", "block"],
        row_start: int = 0,
        row_end: int | None = None,
        column_start: int = 0,
        column_end: int | None = None,
    ) -> SelectionContext:
        """Inspect a row, column or block and return bounded factual selection context."""
        return await service.selection(
            model_id,
            SelectionParameters(
                kind=kind,
                row_start=row_start,
                row_end=row_end,
                column_start=column_start,
                column_end=column_end,
            ),
        )

    @register()
    async def get_optimization_families(
        model_id: str,
        max_families: Annotated[int, Field(ge=1, le=200, strict=True)] = 30,
    ) -> ModelFamilies:
        """Group rows and columns into name-based families with exact statistics.

        The grouping (text before the first '[' or '(', otherwise the name without a
        trailing index) is a HEURISTIC; every count, sense, domain, range and
        family-block nonzero count inside it is exact. Returns the max_families
        largest families per axis in model order and the densest blocks between them;
        totals and truncation flags describe the whole model.
        """
        return await service.families(model_id, FamilyParameters(max_families=max_families))

    @register(read_only=False)
    async def run_optimization_model(
        model_id: str,
        time_limit_seconds: Annotated[
            float, Field(ge=0, le=MAX_TIME_LIMIT_SECONDS, strict=True)
        ] = DEFAULT_TIME_LIMIT_SECONDS,
        mip_relative_gap: Annotated[float, Field(ge=0, le=1, strict=True)] = 0.0001,
        threads: Annotated[int, Field(ge=1, le=4, strict=True)] = 1,
        presolve: Literal["choose", "on", "off"] = "choose",
        mip_heuristic_effort: Annotated[
            float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)
        ] = DEFAULT_MIP_HEURISTIC_EFFORT,
        mip_detect_symmetry: Annotated[bool, Field(strict=True)] = True,
        mip_allow_restart: Annotated[bool, Field(strict=True)] = True,
        simplex_strategy: SimplexStrategy = "dual",
        random_seed: Annotated[int, Field(ge=0, le=MAX_RANDOM_SEED, strict=True)] = 0,
    ) -> RunRecord:
        """Solve a stored model once, independently validate its solution, and record the run.

        Allowlisted search options never change the model or its tolerances and
        default to HiGHS defaults: presolve, mip_heuristic_effort [0, 1],
        mip_detect_symmetry, mip_allow_restart, simplex_strategy; threads is 1-4.
        Runs with different time limit, MIP gap or random_seed are not
        comparable. One runtime is an observation, not a benchmark.
        """
        return await service.solve(
            model_id,
            RunParameters(
                time_limit_seconds=time_limit_seconds,
                mip_relative_gap=mip_relative_gap,
                threads=threads,
                random_seed=random_seed,
                presolve=presolve,
                mip_heuristic_effort=mip_heuristic_effort,
                mip_detect_symmetry=mip_detect_symmetry,
                mip_allow_restart=mip_allow_restart,
                simplex_strategy=simplex_strategy,
            ),
        )

    @register()
    async def explain_optimization_conflict(
        model_id: str,
        time_limit_seconds: Annotated[
            float, Field(ge=0, le=MAX_TIME_LIMIT_SECONDS, strict=True)
        ] = DEFAULT_TIME_LIMIT_SECONDS,
    ) -> ConflictReport:
        """Extract and independently verify a native HiGHS conflict (IIS).

        For a MIP whose LP relaxation is infeasible, the conflict comes from the
        relaxation and certifies MIP infeasibility; irreducibility then refers to
        the relaxation only. An integer-only infeasibility is reported unsupported.
        Membership comes from the solver and subsystem verification, never from a
        language model.
        """
        return await service.conflict(
            model_id, ConflictParameters(time_limit_seconds=time_limit_seconds)
        )

    @register()
    async def list_optimization_runs(model_id: str) -> RunList:
        """List this temporary model's recorded solves, metrics and validation results."""
        return service.list_runs(model_id)

    @register()
    async def compare_optimization_runs(
        baseline_run_id: str, candidate_run_id: str
    ) -> RunComparison:
        """Compare recorded runs with explicit comparability and validation checks."""
        return service.compare(baseline_run_id, candidate_run_id)

    @register(read_only=False)
    async def start_optimization_experiment(
        model_id: str,
        max_runs: Annotated[int, Field(ge=1, le=10, strict=True)] = 3,
        time_limit_seconds: Annotated[
            float, Field(ge=0, le=MAX_TIME_LIMIT_SECONDS, strict=True)
        ] = DEFAULT_TIME_LIMIT_SECONDS,
        repeats: Annotated[int, Field(ge=1, le=MAX_REPEATS, strict=True)] = 1,
    ) -> ExperimentSession:
        """Start a budgeted parameter-only experiment and solve its baseline.

        max_runs INCLUDES the baseline. File semantics cannot change. Each trial
        needs one hypothesis. With repeats k (1-5), the baseline and every trial
        run k validated solves with seeds 0..k-1 and record ONE run (one budget
        attempt), so each can take about k x time_limit_seconds of wall time.
        No automatic performance guarantee or benchmark claim is inferred.
        """
        return await service.start_experiment(
            StartExperimentParameters(
                model_id=model_id,
                max_runs=max_runs,
                time_limit_seconds=time_limit_seconds,
                repeats=repeats,
            )
        )

    @register()
    async def get_optimization_experiment(experiment_id: str) -> ExperimentSession:
        """Read remaining attempt budget, retained run and any failed trial."""
        return service.session(experiment_id)

    @register(read_only=False)
    async def run_optimization_trial(
        experiment_id: str,
        hypothesis: Annotated[str, Field(min_length=1, max_length=1000)],
        presolve: Literal["choose", "on", "off"] = "choose",
        threads: Annotated[int, Field(ge=1, le=4, strict=True)] = 1,
        mip_heuristic_effort: Annotated[
            float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)
        ] = DEFAULT_MIP_HEURISTIC_EFFORT,
        mip_detect_symmetry: Annotated[bool, Field(strict=True)] = True,
        mip_allow_restart: Annotated[bool, Field(strict=True)] = True,
        simplex_strategy: SimplexStrategy = "dual",
    ) -> TrialResult:
        """Test ONE solver-configuration hypothesis within the session budget.

        Allowlist: threads (1-4) and search options defaulting to HiGHS defaults:
        presolve, mip_heuristic_effort [0, 1], mip_detect_symmetry,
        mip_allow_restart, simplex_strategy (choose, dual or primal). Model bytes, time
        budget, MIP gap, seeds and repeats stay fixed. The run is independently
        validated, compared to the retained parent, and kept or rejected. With
        repeats, a runtime win needs non-overlapping runtime ranges across seeds
        and seed disagreement is inconclusive; neither is a benchmark.
        Failed/interrupted launched trials also consume an attempt.
        """
        return await service.trial(
            experiment_id,
            TrialParameters(
                hypothesis=hypothesis,
                presolve=presolve,
                threads=threads,
                mip_heuristic_effort=mip_heuristic_effort,
                mip_detect_symmetry=mip_detect_symmetry,
                mip_allow_restart=mip_allow_restart,
                simplex_strategy=simplex_strategy,
            ),
        )

    @register(read_only=False)
    async def cancel_optimization_experiment(experiment_id: str) -> ExperimentSession:
        """Cancel a budgeted experiment and terminate an active trial."""
        return await service.cancel(experiment_id)

    return tools
