"""One bounded application service for local HTTP and MCP Apps."""

import asyncio
import os
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from server.core.errors import ModelError
from server.core.experiments import compare_runs, describe_trial_options, make_run_record
from server.core.families import limit_families
from server.core.files import MAX_FILE_BYTES, download_file, model_filename
from server.core.limits import TRANSFER_TIMEOUT_SECONDS
from server.core.runner import run_model
from server.core.workspace import ModelEntry, Workspace
from server.schemas.conflicts import ConflictParameters, ConflictReport
from server.schemas.diagnostics import ModelAnalysis
from server.schemas.experiments import (
    ExperimentSession,
    RunComparison,
    RunRecord,
    TrialResult,
    ValidatedSolveParameters,
    ValidatedSolveResult,
)
from server.schemas.families import FamilyParameters, ModelFamilies
from server.schemas.files import UploadedFile
from server.schemas.matrix import (
    MatrixParameters,
    MatrixView,
    SelectionContext,
    SelectionParameters,
)
from server.schemas.sources import SourcesParameters, SourcesReport
from server.schemas.workspace import (
    DeleteResult,
    Example,
    ExampleList,
    RunList,
    RunParameters,
    StartExperimentParameters,
    TrialParameters,
    WorkspaceOverview,
)

MAX_RUNS = 100
MAX_EXPERIMENTS = 20
EXAMPLES = ExampleList(
    examples=[
        Example(
            name="tiny.lp", label="Small LP", description="One variable, a known optimum of 7."
        ),
        Example(name="mip.lp", label="Mixed integer", description="A small maximization model."),
        Example(
            name="possible_big_m.lp",
            label="Possible Big-M",
            description="A large binary coefficient.",
        ),
        Example(
            name="infeasible.lp",
            label="Infeasible LP",
            description="Network demand exceeds plant capacity.",
        ),
        Example(
            name="disconnected.mps",
            label="Disconnected MPS",
            description="Independent graph components.",
        ),
    ]
)


class WorkbenchService:
    def __init__(self) -> None:
        self.workspace = Workspace()
        self.slots = asyncio.Semaphore(2)
        self.runs: dict[str, RunRecord] = {}
        self.sessions: dict[str, ExperimentSession] = {}
        self._reserved_runs = 0
        self._reserved_sessions = 0
        self._trial_tasks: dict[str, asyncio.Task] = {}
        self._active_tasks: set[asyncio.Task[object]] = set()
        self._selections: dict[str, SelectionContext] = {}
        self._closed = False

    @asynccontextmanager
    async def capacity(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=0.1)
        except TimeoutError as exc:
            raise ModelError("server_busy", "Both model workers are busy; retry later.") from exc
        try:
            task = asyncio.current_task()
            if task is not None:
                self._active_tasks.add(task)
            yield
        finally:
            if task is not None:
                self._active_tasks.discard(task)
            self.slots.release()

    @asynccontextmanager
    async def model_job(self, model_id: str) -> AsyncIterator[ModelEntry]:
        async with self.capacity():
            entry = self.workspace.get(model_id)
            entry.busy += 1
            try:
                yield entry
            finally:
                entry.busy -= 1

    async def _inspect_entry(self, entry: ModelEntry) -> WorkspaceOverview:
        entry.busy = 1
        try:
            result = await run_model("overview", entry.path, entry.summary.name)
            entry.analysis = ModelAnalysis.model_validate(result)
            if entry.analysis.inspection.model_hash != entry.summary.model_hash:
                raise ModelError("model_changed", "Model snapshot hash does not match.")
            entry.busy = 0
            return self.overview(entry.summary.model_id)
        except BaseException:
            entry.busy = 0
            self.workspace.delete(entry.summary.model_id)
            raise

    async def open_data(self, name: str, data: bytes) -> WorkspaceOverview:
        async with self.capacity():
            return await self._inspect_entry(self.workspace.add(name, data))

    async def open_stream(self, name: str, chunks: AsyncIterator[bytes]) -> WorkspaceOverview:
        name = model_filename(UploadedFile(download_url="local", file_id="local", file_name=name))
        async with self.capacity():
            with TemporaryDirectory(prefix="or-lens-upload-") as folder:
                path = Path(folder) / ("model" + Path(name).suffix.lower())
                size = 0
                try:
                    async with asyncio.timeout(TRANSFER_TIMEOUT_SECONDS):
                        with path.open("wb") as output:
                            async for chunk in chunks:
                                size += len(chunk)
                                if size > MAX_FILE_BYTES:
                                    raise ModelError("file_too_large", "Model file exceeds 1 GiB.")
                                output.write(chunk)
                except TimeoutError as exc:
                    raise ModelError("upload_timeout", "Upload exceeded its time limit.") from exc
                return await self._inspect_entry(self.workspace.add_file(name, path))

    async def open_upload(self, file: UploadedFile, filename: str | None) -> WorkspaceOverview:
        name = model_filename(file, filename)
        async with self.capacity():
            with TemporaryDirectory(prefix="or-lens-upload-") as folder:
                path = Path(folder) / ("model" + Path(name).suffix.lower())
                await download_file(file, path)
                return await self._inspect_entry(self.workspace.add_file(name, path))

    async def open_local_path(self, raw_path: str) -> WorkspaceOverview:
        """Snapshot a regular local LP/MPS file without following symlinks."""
        source = Path(raw_path)
        if not source.is_absolute():
            raise ModelError("invalid_local_path", "Choose an absolute local model path.")
        try:
            before_link = source.lstat()
            before = source.stat()
        except (OSError, ValueError) as exc:
            raise ModelError("local_file_unavailable", "Local model file is unavailable.") from exc
        if stat.S_ISLNK(before_link.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ModelError("invalid_local_file", "Local model must be a regular file.")
        name = source.name
        try:
            name = model_filename(
                UploadedFile(download_url="local", file_id="local", file_name=name)
            )
        except ModelError as exc:
            raise ModelError(exc.code, exc.message) from None
        if before.st_size > MAX_FILE_BYTES:
            raise ModelError("file_too_large", "Model file exceeds 1 GiB.")
        if before.st_size == 0:
            raise ModelError("empty_file", "Model file is empty.")

        async with self.capacity():
            staged = self.workspace.root / (uuid4().hex + Path(name).suffix.lower() + ".part")
            copied = 0
            try:
                async with asyncio.timeout(TRANSFER_TIMEOUT_SECONDS):
                    descriptor = os.open(
                        source, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
                    )
                    with os.fdopen(descriptor, "rb") as input_file, staged.open("xb") as output:
                        opened = os.fstat(input_file.fileno())
                        if not stat.S_ISREG(opened.st_mode):
                            raise ModelError(
                                "invalid_local_file", "Local model must be a regular file."
                            )
                        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
                            raise ModelError("model_changed", "Local model changed before copying.")
                        while chunk := input_file.read(64 * 1024):
                            copied += len(chunk)
                            if copied > MAX_FILE_BYTES:
                                raise ModelError("file_too_large", "Model file exceeds 1 GiB.")
                            output.write(chunk)
                            await asyncio.sleep(0)
                after = source.stat()
                stable = (
                    after.st_dev == before.st_dev
                    and after.st_ino == before.st_ino
                    and after.st_size == before.st_size
                    and after.st_mtime_ns == before.st_mtime_ns
                    and after.st_ctime_ns == before.st_ctime_ns
                    and copied == before.st_size
                )
                if not stable:
                    raise ModelError("model_changed", "Local model changed while it was copied.")
                return await self._inspect_entry(self.workspace.add_file(name, staged))
            except TimeoutError as exc:
                staged.unlink(missing_ok=True)
                raise ModelError(
                    "upload_timeout", "Local file copy exceeded its time limit."
                ) from exc
            except OSError as exc:
                staged.unlink(missing_ok=True)
                raise ModelError(
                    "local_file_unavailable", "Local model file is unavailable."
                ) from exc
            except BaseException:
                staged.unlink(missing_ok=True)
                raise

    async def open_example(self, name: str) -> WorkspaceOverview:
        if name not in {example.name for example in EXAMPLES.examples}:
            raise ModelError("example_not_found", "Unknown example model.")
        path = Path(__file__).parent / "examples" / name
        return await self.open_data(name, path.read_bytes())

    def overview(self, model_id: str) -> WorkspaceOverview:
        entry = self.workspace.get(model_id)
        if entry.analysis is None:
            raise ModelError("model_busy", "Model inspection is still running.")
        history = self.list_runs(model_id).runs
        return WorkspaceOverview(
            model_id=model_id,
            inspection=entry.analysis.inspection,
            analysis=entry.analysis,
            latest_run=history[-1] if history else None,
            latest_conflict=entry.conflict,
        )

    def delete(self, model_id: str) -> DeleteResult:
        if any(
            session.model_id == model_id and experiment_id in self._trial_tasks
            for experiment_id, session in self.sessions.items()
        ):
            raise ModelError("model_busy", "Wait for this model's trial to finish or cancel it.")
        self.workspace.delete(model_id)
        self._selections.pop(model_id, None)
        self.runs = {key: run for key, run in self.runs.items() if run.model_id != model_id}
        self.sessions = {
            key: session for key, session in self.sessions.items() if session.model_id != model_id
        }
        return DeleteResult(deleted=True)

    async def matrix(self, model_id: str, parameters: MatrixParameters) -> MatrixView:
        async with self.model_job(model_id) as entry:
            return MatrixView.model_validate(
                await run_model("matrix", entry.path, entry.summary.name, parameters.model_dump())
            )

    async def selection(self, model_id: str, parameters: SelectionParameters) -> SelectionContext:
        async with self.model_job(model_id) as entry:
            selection = SelectionContext.model_validate(
                await run_model(
                    "selection", entry.path, entry.summary.name, parameters.model_dump()
                )
            )
            self._selections[model_id] = selection
            return selection

    def last_selection(self, model_id: str) -> SelectionContext | None:
        self.workspace.get(model_id)
        return self._selections.get(model_id)

    async def families(self, model_id: str, parameters: FamilyParameters) -> ModelFamilies:
        """Name-based families; the deterministic summary is computed once per snapshot."""
        summary = self.workspace.get(model_id).families
        if summary is None:
            async with self.model_job(model_id) as entry:
                summary = ModelFamilies.model_validate(
                    await run_model("families", entry.path, entry.summary.name)
                )
                if summary.model_hash != entry.summary.model_hash:
                    raise ModelError("model_changed", "Model snapshot hash does not match.")
                entry.families = summary
        return limit_families(summary, parameters.max_families)

    async def conflict(self, model_id: str, parameters: ConflictParameters) -> ConflictReport:
        """Extract a conflict; the latest report is kept with the snapshot for the overview."""
        async with self.model_job(model_id) as entry:
            report = ConflictReport.model_validate(
                await run_model("conflict", entry.path, entry.summary.name, parameters.model_dump())
            )
            if report.model_hash != entry.summary.model_hash:
                raise ModelError("model_changed", "Model snapshot hash does not match.")
            entry.conflict = report
            return report

    async def sources(self, model_id: str, parameters: SourcesParameters) -> SourcesReport:
        """Find every independent source of infeasibility with STOLP."""
        async with self.model_job(model_id) as entry:
            report = SourcesReport.model_validate(
                await run_model("sources", entry.path, entry.summary.name, parameters.model_dump())
            )
            if report.model_hash != entry.summary.model_hash:
                raise ModelError("model_changed", "Model snapshot hash does not match.")
            return report

    async def _solve_validated(
        self, entry: ModelEntry, parameters: ValidatedSolveParameters
    ) -> ValidatedSolveResult:
        # One worker call per solve keeps the time limit + 5 s worker deadline per repeat.
        result = ValidatedSolveResult.model_validate(
            await run_model(
                "solve_validated", entry.path, entry.summary.name, parameters.model_dump()
            )
        )
        if result.result.model_hash != entry.summary.model_hash:
            raise ModelError("model_changed", "Model snapshot hash does not match.")
        return result

    async def _run(
        self,
        entry: ModelEntry,
        parameters: RunParameters,
        description: str,
        parent_run_id: str | None = None,
        hypothesis: str | None = None,
        repeats: int = 1,
    ) -> RunRecord:
        """Record one run from `repeats` sequential validated solves.

        Seeds are parameters.random_seed + 0..repeats-1. Any failed repeat fails
        the run; cancellation stops before the next repeat is launched.
        """
        if len(self.runs) + self._reserved_runs >= MAX_RUNS:
            raise ModelError("run_limit", "Run history is full. Remove a model to free its runs.")
        self._reserved_runs += 1
        try:
            base = parameters.validated_parameters()
            outcomes: list[ValidatedSolveResult] = []
            for offset in range(repeats):
                seeded = ValidatedSolveParameters.model_validate(
                    {
                        **base.model_dump(),
                        "solve": {
                            **base.solve.model_dump(),
                            "random_seed": base.solve.random_seed + offset,
                        },
                    }
                )
                outcomes.append(await self._solve_validated(entry, seeded))
            run = make_run_record(
                entry.summary.model_id, parent_run_id, description, outcomes, hypothesis=hypothesis
            )
            self.runs[run.run_id] = run
            return run
        finally:
            self._reserved_runs -= 1

    async def solve(self, model_id: str, parameters: RunParameters) -> RunRecord:
        async with self.model_job(model_id) as entry:
            return await self._run(entry, parameters, "Manual solver run")

    def list_runs(self, model_id: str) -> RunList:
        self.workspace.get(model_id)
        return RunList(runs=[run for run in self.runs.values() if run.model_id == model_id])

    def compare(self, baseline_run_id: str, candidate_run_id: str) -> RunComparison:
        try:
            baseline, candidate = self.runs[baseline_run_id], self.runs[candidate_run_id]
        except KeyError as exc:
            raise ModelError("run_not_found", "One of the requested runs is unavailable.") from exc
        sense = self.overview(baseline.model_id).inspection.objective.sense
        return compare_runs(baseline, candidate, sense)

    async def start_experiment(self, parameters: StartExperimentParameters) -> ExperimentSession:
        if len(self.sessions) + self._reserved_sessions >= MAX_EXPERIMENTS:
            raise ModelError("experiment_limit", "Experiment history is full. Remove a model.")
        self._reserved_sessions += 1
        try:
            async with self.model_job(parameters.model_id) as entry:
                run = await self._run(
                    entry,
                    RunParameters(time_limit_seconds=parameters.time_limit_seconds),
                    "Experiment baseline",
                    repeats=parameters.repeats,
                )
            session = ExperimentSession(
                experiment_id=uuid4().hex,
                model_id=parameters.model_id,
                max_runs=parameters.max_runs,
                time_limit_seconds=parameters.time_limit_seconds,
                repeats=parameters.repeats,
                baseline_run_id=run.run_id,
                best_run_id=run.run_id,
                run_ids=[run.run_id],
                status="complete" if parameters.max_runs == 1 else "active",
                remaining_runs=parameters.max_runs - 1,
                attempt_count=1,
                created_at=datetime.now(UTC),
            )
            self.sessions[session.experiment_id] = session
            return session.model_copy(deep=True)
        finally:
            self._reserved_sessions -= 1

    def session(self, experiment_id: str) -> ExperimentSession:
        if experiment_id not in self.sessions:
            raise ModelError("experiment_not_found", "Experiment is unavailable.")
        return self.sessions[experiment_id].model_copy(deep=True)

    async def trial(self, experiment_id: str, parameters: TrialParameters) -> TrialResult:
        session = self.session(experiment_id)
        if session.status != "active" or session.remaining_runs <= 0:
            raise ModelError("experiment_closed", "Experiment has no remaining active budget.")
        if experiment_id in self._trial_tasks:
            raise ModelError("experiment_busy", "A trial is already running for this experiment.")
        if not parameters.hypothesis.strip():
            raise ModelError("missing_hypothesis", "Write one hypothesis for this trial.")
        task = asyncio.current_task()
        assert task is not None
        self._trial_tasks[experiment_id] = task
        charged = False
        try:
            async with self.model_job(session.model_id) as entry:
                session.attempt_count += 1
                session.remaining_runs = session.max_runs - session.attempt_count
                session.last_error = None
                self.sessions[experiment_id] = session
                charged = True
                # The session fixes time limit, MIP gap, seed schedule and repeats.
                run = await self._run(
                    entry,
                    RunParameters(
                        time_limit_seconds=session.time_limit_seconds,
                        threads=parameters.threads,
                        presolve=parameters.presolve,
                        mip_heuristic_effort=parameters.mip_heuristic_effort,
                        mip_detect_symmetry=parameters.mip_detect_symmetry,
                        mip_allow_restart=parameters.mip_allow_restart,
                        simplex_strategy=parameters.simplex_strategy,
                    ),
                    describe_trial_options(parameters, parameters.threads),
                    parent_run_id=session.best_run_id,
                    hypothesis=parameters.hypothesis.strip(),
                    repeats=session.repeats,
                )
                comparison = self.compare(session.best_run_id, run.run_id)
                kept = comparison.verdict == "candidate_retained"
                session.run_ids.append(run.run_id)
                if kept:
                    session.best_run_id = run.run_id
                if session.remaining_runs == 0:
                    session.status = "complete"
                self.sessions[experiment_id] = session
                return TrialResult(
                    session=session.model_copy(deep=True), run=run, comparison=comparison, kept=kept
                )
        except BaseException as exc:
            if charged and (current := self.sessions.get(experiment_id)) is not None:
                current.failed_attempts += 1
                current.last_error = (
                    exc.message if isinstance(exc, ModelError) else "Trial interrupted."
                )
                if current.remaining_runs == 0 and current.status != "cancelled":
                    current.status = "complete"
            raise
        finally:
            self._trial_tasks.pop(experiment_id, None)

    async def cancel(self, experiment_id: str) -> ExperimentSession:
        self.session(experiment_id)
        session = self.sessions[experiment_id]
        if session.status == "active":
            session.status = "cancelled"
        task = self._trial_tasks.get(experiment_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.session(experiment_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.cancel_active_operations()
        self.workspace.close()
        self._selections.clear()

    async def cancel_active_operations(self) -> None:
        """Cancel in-flight work so subprocess cleanup runs before workspace removal."""
        current = asyncio.current_task()
        tasks = list({*self._trial_tasks.values(), *self._active_tasks} - {current})
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
