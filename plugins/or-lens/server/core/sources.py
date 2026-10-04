"""Every independent source of infeasibility, found with the STOLP library.

STOLP searches minimal combinations of constraint groups, then the constraints
inside them, and checks fixes on the whole model with the other sources switched
off. Membership and fixes come from solver checks; this module only bounds and
reshapes STOLP's report.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

from server.schemas.sources import (
    InfeasibilitySource,
    SourceChange,
    SourceFix,
    SourcesParameters,
    SourcesReport,
)

SOURCE_LIMIT = 50
GROUP_LIMIT = 20
FIX_LIMIT = 6
CHANGE_LIMIT = 3
NAME_LIMIT = 200
WHERE_LIMIT = 500
SWITCHED_OFF_LIMIT = 100
SUMMARY_LIMIT = 20000

# Measured in STOLP's agent benchmark: agents with this guidance found the wrong
# data more often and used fewer tokens than agents with the bare library.
GUIDANCE = """\
stolp has already searched the whole model. Your job is to interpret the result, not to repeat
the search.

1. Causes. Each numbered source is one independent cause of infeasibility: report exactly that
   many causes. A fix that changes several constraints of one group is still one cause. Do not
   run a solver IIS or another search to confirm the list; to check it, switch off the
   constraints the report names as switched off and re-solve.
2. Fixes are alternatives, not a diagnosis. Each line under "checked on the whole model" makes
   the whole model feasible on its own (other sources switched off). The lines are ordered by
   the size of the change, which says nothing about which data are wrong. stolp cannot know
   which value is wrong: you decide.
3. Decide which data are wrong. For each alternative, read the current values (search the model
   file for the row names) and compare them with comparable data: the same constraint family at
   other indices (other periods, sites, products) and other constraints on the same variables. A
   wrong value is usually the one that is inconsistent with its neighbours, while correct values
   agree with each other. If the model's source code or documentation is available, check what
   each constraint family means: input data, a hard rule, or a definition or balance.
   Definitions and balances (right-hand side 0) rarely hold data errors. State the evidence for
   your choice; if the data cannot decide, present the alternatives neutrally.
4. Indices. A fix is computed for one representative index of a source; the same kind of change
   is needed at every index listed for that source.
5. Verify. Apply the chosen change in memory (for example highspy changeRowBounds on the named
   rows) at all indices and re-solve the whole model; for a MIP, check the LP relaxation first.
"""

MESSAGES = {
    "feasible": "The model is feasible.",
    "unknown": "The solver did not decide feasibility within the time limit.",
    "unavailable": "The stolp package is not installed in the OR Lens environment.",
}


def _bounded(text: str, limit: int) -> tuple[str, bool]:
    return (text[:limit], True) if len(text) > limit else (text, False)


@contextmanager
def _native_stdout_silenced() -> Iterator[None]:
    """Send file descriptor 1 to /dev/null: stdout carries the worker's JSON protocol."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def _change(fix: Any) -> SourceChange | None:
    values = (float(fix.current), float(fix.needed), float(fix.shift))
    if not all(math.isfinite(value) for value in values):
        return None
    name, truncated = _bounded(str(fix.constraint.label), NAME_LIMIT)
    return SourceChange(
        constraint=name,
        constraint_truncated=truncated,
        bound=fix.bound,
        current=values[0],
        needed=values[1],
        shift=values[2],
    )


def _fix(group: str, fixes: list[Any], checked: bool) -> SourceFix | None:
    changes = [change for change in (_change(fix) for fix in fixes) if change is not None]
    if not changes or len(changes) > CHANGE_LIMIT:
        return None
    name, truncated = _bounded(group, NAME_LIMIT)
    return SourceFix(
        group=name, group_truncated=truncated, changes=changes, checked_on_whole_model=checked
    )


def _source(found: Any) -> InfeasibilitySource:
    fixes = [_fix(fix.group, fix.changes, True) for fix in found.fixes]
    if not found.fixes and found.example is not None:
        fixes = [_fix(found.example.constraint.group, [found.example], False)]
    kept = [fix for fix in fixes if fix is not None]
    where, where_truncated = _bounded(str(found.where), WHERE_LIMIT)
    groups = [_bounded(str(group), NAME_LIMIT)[0] for group in found.groups]
    carried = found.carried_from
    return InfeasibilitySource(
        groups=groups[:GROUP_LIMIT],
        groups_truncated=len(groups) > GROUP_LIMIT,
        where=where,
        where_truncated=where_truncated,
        conflicts=max(1, len(found.conflicts)),
        fixes=kept[:FIX_LIMIT],
        fixes_truncated=len(kept) > FIX_LIMIT,
        carried_from=None
        if carried is None
        else [_bounded(str(group), NAME_LIMIT)[0] for group in carried][:GROUP_LIMIT],
    )


def sources_report(path: Path, model_hash: str, parameters: SourcesParameters) -> SourcesReport:
    """Run STOLP on the model file and return its sources within this module's bounds."""
    started = time.monotonic()
    try:
        from stolp import find_conflicts
    except ImportError:
        return SourcesReport(
            model_hash=model_hash,
            status="unavailable",
            message=MESSAGES["unavailable"],
            runtime_seconds=time.monotonic() - started,
        )
    with _native_stdout_silenced():
        report = find_conflicts(
            str(path),
            solver="highs",
            mode="groups",
            detail=True,
            max_time=parameters.time_limit_seconds,
        )
        summary, summary_truncated = _bounded(report.summary(), SUMMARY_LIMIT)
        found = report.sources()
    status = report.status.value
    independent = [source for source in found if source.carried_from is None]
    switched = [ref.label for ref in report.repair] or list(report.repair_groups)
    if status == "infeasible":
        count = len(independent)
        message = (
            f"{count} independent source{'' if count == 1 else 's'} of infeasibility."
            if found
            else "The model is infeasible; see the summary for the group conflicts."
        )
        if report.stopped:
            message += " The search stopped early, so the list may be incomplete."
    elif status == "unknown" and report.stopped == "max_time":
        message = (
            "The time limit ended before feasibility was decided; increase time_limit_seconds."
        )
    else:
        message = MESSAGES[status]
    return SourcesReport(
        model_hash=model_hash,
        status=status,
        stolp_version=metadata.version("stolp"),
        message=message,
        source_count=len(independent),
        sources=[_source(source) for source in found[:SOURCE_LIMIT]],
        sources_truncated=len(found) > SOURCE_LIMIT,
        switched_off=[_bounded(str(name), NAME_LIMIT)[0] for name in switched[:SWITCHED_OFF_LIMIT]],
        switched_off_total=len(switched),
        summary=summary,
        summary_truncated=summary_truncated,
        guidance=GUIDANCE if status == "infeasible" else "",
        stopped=report.stopped,
        solves=report.nsolves,
        solver_seconds=max(0.0, float(report.solve_time)),
        runtime_seconds=time.monotonic() - started,
    )
