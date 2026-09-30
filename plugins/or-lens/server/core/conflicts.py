"""Bounded native HiGHS IIS extraction with independent verification.

Continuous LPs are analyzed directly. A MIP is analyzed through its LP relaxation:
an IIS is extracted from and verified on that relaxation; conflicts that need
integrality are reported as unsupported.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Literal

import highspy
import numpy as np

from server.core.model_loader import LoadedModel
from server.schemas.conflicts import (
    ConflictColumn,
    ConflictEdge,
    ConflictGroup,
    ConflictMember,
    ConflictMetadata,
    ConflictParameters,
    ConflictReport,
)

ROW_LIMIT = 100
BOUND_LIMIT = 100
EDGE_LIMIT = 300
GROUP_LIMIT = 20
NAME_LIMIT = 200
IIS_STRATEGY = int(highspy.IisStrategy.kIisStrategyFromLp) | int(
    highspy.IisStrategy.kIisStrategyIrreducible
)
IIS_STATUS_TIME_LIMIT = 1
IIS_STATUS_REDUCIBLE = 2
IIS_STATUS_IRREDUCIBLE = 3
LP_METHOD = "highs_iis_from_lp_irreducible"
RELAXATION_METHOD = "highs_iis_lp_relaxation"


def _is_mip(model: LoadedModel) -> bool:
    return any(
        variable_type == highspy.HighsVarType.kInteger for variable_type in model.lp.integrality_
    )


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _msg(relax: bool, lp_message: str, relaxation_message: str) -> str:
    return relaxation_message if relax else lp_message


def _new_highs(model: LoadedModel, *, relax: bool = False) -> highspy.Highs:
    highs = highspy.Highs()
    # HiGHS has process-global scheduler state. Keep direct in-process calls
    # compatible with the project's deterministic single-thread solve default.
    _require_ok(highs.setOptionValue("threads", 1))
    _require_ok(highs.setOptionValue("output_flag", False))
    _require_ok(highs.setOptionValue("log_to_console", False))
    # passModel copies the LP: later changes never reach model.lp or model.highs.
    if highs.passModel(model.lp) == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS could not load the LP conflict model.")
    if relax:
        _relax_integrality(highs, model.lp.num_col_)
    return highs


def _relax_integrality(highs: highspy.Highs, num_col: int) -> None:
    """Make every column of this private copy continuous: the LP relaxation."""
    if num_col == 0:
        return
    status = highs.changeColsIntegrality(
        num_col,
        np.arange(num_col, dtype=np.int32),
        np.full(num_col, int(highspy.HighsVarType.kContinuous), dtype=np.uint8),
    )
    if status != highspy.HighsStatus.kOk or any(
        variable_type != highspy.HighsVarType.kContinuous
        for variable_type in highs.getLp().integrality_
    ):
        raise RuntimeError("HiGHS could not form the LP relaxation.")


def _require_ok(status: highspy.HighsStatus) -> None:
    if status == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected conflict configuration.")


def _member_bound(status: int) -> Literal["lower", "upper", "both"] | None:
    if status == int(highspy.IisBoundStatus.kIisBoundStatusLower):
        return "lower"
    if status == int(highspy.IisBoundStatus.kIisBoundStatusUpper):
        return "upper"
    if status == int(highspy.IisBoundStatus.kIisBoundStatusBoxed):
        return "both"
    return None


def _metadata(
    *,
    verified: bool = False,
    irreducible: bool = False,
    total_rows: int = 0,
    total_bounds: int = 0,
    rows_truncated: bool = False,
    bounds_truncated: bool = False,
    edges_truncated: bool = False,
    iis_status: int = 0,
    iis_valid: bool = False,
    validation_status: str = "not_run",
    relaxation: bool = False,
) -> ConflictMetadata:
    return ConflictMetadata(
        verified=verified,
        irreducible=irreducible,
        total_rows=total_rows,
        total_bounds=total_bounds,
        rows_truncated=rows_truncated,
        bounds_truncated=bounds_truncated,
        edges_truncated=edges_truncated,
        iis_status=iis_status,
        iis_valid=iis_valid,
        validation_status=validation_status,
        relaxation=relaxation,
    )


def _report(
    model: LoadedModel,
    started: float,
    status: str,
    message: str,
    *,
    method: str,
    relaxation: bool = False,
    metadata: ConflictMetadata | None = None,
) -> ConflictReport:
    return ConflictReport(
        model_hash=model.model_hash,
        status=status,  # type: ignore[arg-type]
        method=method,
        message=message,
        conflict_metadata=(metadata or _metadata()).model_copy(update={"relaxation": relaxation}),
        runtime_seconds=max(0.0, time.monotonic() - started),
    )


def _remaining(started: float, limit: float) -> float:
    return max(0.0, limit - (time.monotonic() - started))


def _has_feasible_primal(highs: highspy.Highs) -> bool:
    solution = highs.getSolution()
    info = highs.getInfo()
    return bool(
        info.valid
        and solution.value_valid
        and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
    )


def _feasible_status(highs: highspy.Highs) -> bool:
    status = highs.getModelStatus()
    if status in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kUnbounded):
        return True
    return _has_feasible_primal(highs)


def _bounded_name(name: str) -> tuple[str, bool]:
    return name[:NAME_LIMIT], len(name) > NAME_LIMIT


def _groups(names: list[str], members: list[ConflictMember]) -> list[ConflictGroup]:
    """Group only names with an explicit bracketed family prefix, e.g. demand[1]."""
    prefixes = Counter(
        names[member.index].split("[", 1)[0]
        for member in members
        if "[" in names[member.index] and names[member.index].split("[", 1)[0]
    )
    groups: list[ConflictGroup] = []
    for prefix, count in sorted(prefixes.items(), key=lambda item: (-item[1], item[0])):
        if count <= 1:
            continue
        display_prefix, truncated = _bounded_name(prefix)
        groups.append(ConflictGroup(prefix=display_prefix, prefix_truncated=truncated, count=count))
        if len(groups) == GROUP_LIMIT:
            break
    return groups


def _members(model: LoadedModel, iis: Any) -> tuple[list[ConflictMember], list[ConflictMember]]:
    rows: list[ConflictMember] = []
    bounds: list[ConflictMember] = []
    for index, value in zip(iis.row_index_, iis.row_bound_, strict=True):
        bound = _member_bound(int(value))
        if bound is None:
            continue
        name, name_truncated = _bounded_name(model.row_names[int(index)])
        rows.append(
            ConflictMember(
                index=int(index),
                name=name,
                name_truncated=name_truncated,
                bound=bound,
                lower=_finite(model.lp.row_lower_[int(index)]),
                upper=_finite(model.lp.row_upper_[int(index)]),
            )
        )
    for index, value in zip(iis.col_index_, iis.col_bound_, strict=True):
        bound = _member_bound(int(value))
        if bound is None:
            continue
        name, name_truncated = _bounded_name(model.column_names[int(index)])
        bounds.append(
            ConflictMember(
                index=int(index),
                name=name,
                name_truncated=name_truncated,
                bound=bound,
                lower=_finite(model.lp.col_lower_[int(index)]),
                upper=_finite(model.lp.col_upper_[int(index)]),
            )
        )
    return rows, bounds


def _edges(model: LoadedModel, rows: list[ConflictMember]) -> tuple[list[ConflictEdge], bool]:
    edges: list[ConflictEdge] = []
    for row in rows:
        start, end = model.matrix.indptr[row.index : row.index + 2]
        for column in model.matrix.indices[start:end]:
            if len(edges) == EDGE_LIMIT:
                return edges, True
            edges.append(ConflictEdge(row_index=row.index, column_index=int(column)))
    return edges, False


def _edge_columns(model: LoadedModel, edges: list[ConflictEdge]) -> list[ConflictColumn]:
    columns: list[ConflictColumn] = []
    for index in dict.fromkeys(edge.column_index for edge in edges):
        name, name_truncated = _bounded_name(model.column_names[index])
        columns.append(ConflictColumn(index=index, name=name, name_truncated=name_truncated))
    return columns


def _set_selected_bounds(
    highs: highspy.Highs,
    model: LoadedModel,
    rows: list[ConflictMember],
    bounds: list[ConflictMember],
) -> None:
    infinity = highs.getInfinity()
    selected_rows = {member.index: member for member in rows}
    selected_bounds = {member.index: member for member in bounds}
    for index in range(model.lp.num_row_):
        member = selected_rows.get(index)
        lower = (
            model.lp.row_lower_[index]
            if member and member.bound in ("lower", "both")
            else -infinity
        )
        upper = (
            model.lp.row_upper_[index] if member and member.bound in ("upper", "both") else infinity
        )
        _require_ok(highs.changeRowBounds(index, lower, upper))
    for index in range(model.lp.num_col_):
        member = selected_bounds.get(index)
        lower = (
            model.lp.col_lower_[index]
            if member and member.bound in ("lower", "both")
            else -infinity
        )
        upper = (
            model.lp.col_upper_[index] if member and member.bound in ("upper", "both") else infinity
        )
        _require_ok(highs.changeColBounds(index, lower, upper))
        _require_ok(highs.changeColCost(index, 0.0))


def _verify(
    model: LoadedModel,
    rows: list[ConflictMember],
    bounds: list[ConflictMember],
    remaining: float,
    *,
    relax: bool = False,
) -> str:
    """Re-solve only the selected subsystem; ``relax`` must match the IIS source."""
    if remaining <= 0:
        return "time_limit"
    highs = _new_highs(model, relax=relax)
    _set_selected_bounds(highs, model, rows, bounds)
    _require_ok(highs.setOptionValue("time_limit", remaining))
    if highs.run() == highspy.HighsStatus.kError:
        return "error"
    status = highs.getModelStatus()
    if status == highspy.HighsModelStatus.kInfeasible:
        return "infeasible"
    if status == highspy.HighsModelStatus.kTimeLimit:
        return "time_limit"
    return status.name


def _mip_feasibility_report(
    model: LoadedModel, started: float, limit: float, method: str
) -> ConflictReport:
    """Classify a MIP whose LP relaxation HiGHS found feasible."""
    highs = _new_highs(model)
    remaining = _remaining(started, limit)
    if remaining <= 0:
        return _report(
            model,
            started,
            "time_limit",
            "Conflict time budget expired.",
            method=method,
            relaxation=True,
        )
    # Only feasibility is asked: a zero objective keeps the MIP's feasible set, lets
    # HiGHS stop at its first integer-feasible point, and avoids the "unbounded or
    # infeasible" status HiGHS returns for some feasible MIPs with unbounded objective.
    num_col = model.lp.num_col_
    _require_ok(
        highs.changeColsCost(
            num_col, np.arange(num_col, dtype=np.int32), np.zeros(num_col, dtype=np.float64)
        )
    )
    _require_ok(highs.setOptionValue("time_limit", remaining))
    if highs.run() == highspy.HighsStatus.kError:
        return _report(
            model,
            started,
            "unavailable",
            "LP relaxation is feasible; HiGHS could not check MIP feasibility.",
            method=method,
            relaxation=True,
        )
    status = highs.getModelStatus()
    if _has_feasible_primal(highs):
        return _report(
            model,
            started,
            "feasible",
            "Model is feasible; no conflict exists.",
            method=method,
            relaxation=True,
            metadata=_metadata(validation_status="feasible"),
        )
    if status == highspy.HighsModelStatus.kInfeasible:
        return _report(
            model,
            started,
            "unsupported",
            "HiGHS found the MIP infeasible but its LP relaxation feasible, so the conflict "
            "involves integrality. The native LP IIS cannot isolate it; no members are reported.",
            method=method,
            relaxation=True,
            metadata=_metadata(validation_status="integer_infeasible"),
        )
    if status == highspy.HighsModelStatus.kTimeLimit:
        return _report(
            model,
            started,
            "time_limit",
            "LP relaxation is feasible; the MIP feasibility check reached its time limit.",
            method=method,
            relaxation=True,
            metadata=_metadata(validation_status="time_limit"),
        )
    return _report(
        model,
        started,
        "unavailable",
        "LP relaxation is feasible; HiGHS could not establish MIP feasibility.",
        method=method,
        relaxation=True,
        metadata=_metadata(validation_status=status.name),
    )


def _extract(
    model: LoadedModel, started: float, limit: float, *, relax: bool, method: str
) -> ConflictReport:
    """Check feasibility, then extract and verify an IIS of the LP or LP relaxation."""
    highs = _new_highs(model, relax=relax)
    remaining = _remaining(started, limit)
    if remaining <= 0:
        return _report(
            model,
            started,
            "time_limit",
            "Conflict time budget expired.",
            method=method,
            relaxation=relax,
        )
    _require_ok(highs.setOptionValue("time_limit", remaining))
    if highs.run() == highspy.HighsStatus.kError:
        return _report(
            model,
            started,
            "unavailable",
            _msg(
                relax,
                "HiGHS could not check model feasibility.",
                "HiGHS could not check LP relaxation feasibility.",
            ),
            method=method,
            relaxation=relax,
        )
    if _feasible_status(highs):
        if relax:
            del highs  # Release the relaxation before solving the integer model.
            return _mip_feasibility_report(model, started, limit, method)
        return _report(
            model,
            started,
            "feasible",
            "Model is feasible; no conflict exists.",
            method=method,
            metadata=_metadata(validation_status="feasible"),
        )
    if highs.getModelStatus() == highspy.HighsModelStatus.kTimeLimit:
        return _report(
            model,
            started,
            "time_limit",
            _msg(
                relax,
                "Feasibility check reached its time limit.",
                "LP relaxation feasibility check reached its time limit.",
            ),
            method=method,
            relaxation=relax,
            metadata=_metadata(validation_status="time_limit"),
        )
    if highs.getModelStatus() != highspy.HighsModelStatus.kInfeasible:
        return _report(
            model,
            started,
            "unavailable",
            _msg(
                relax,
                "HiGHS could not establish infeasibility.",
                "HiGHS could not establish LP relaxation infeasibility.",
            ),
            method=method,
            relaxation=relax,
            metadata=_metadata(validation_status=highs.getModelStatus().name),
        )

    remaining = _remaining(started, limit)
    if remaining <= 0:
        return _report(
            model,
            started,
            "time_limit",
            "Conflict time budget expired.",
            method=method,
            relaxation=relax,
        )
    _require_ok(highs.setOptionValue("iis_strategy", IIS_STRATEGY))
    # HiGHS uses this IIS-specific option instead of the solve time limit internally.
    _require_ok(highs.setOptionValue("iis_time_limit", remaining))
    status, iis = highs.getIis()
    iis_status = int(iis.status_)
    if iis_status == IIS_STATUS_TIME_LIMIT:
        return _report(
            model,
            started,
            "time_limit",
            _msg(
                relax,
                "IIS extraction reached its time limit.",
                "LP relaxation IIS extraction reached its time limit.",
            ),
            method=method,
            relaxation=relax,
            metadata=_metadata(
                iis_status=iis_status,
                iis_valid=bool(iis.valid_),
                validation_status="time_limit",
            ),
        )
    if (
        status != highspy.HighsStatus.kOk
        or not iis.valid_
        or iis_status not in (IIS_STATUS_REDUCIBLE, IIS_STATUS_IRREDUCIBLE)
    ):
        return _report(
            model,
            started,
            "unavailable",
            _msg(
                relax,
                "HiGHS did not return a valid IIS.",
                "HiGHS did not return a valid LP relaxation IIS.",
            ),
            method=method,
            relaxation=relax,
            metadata=_metadata(
                iis_status=iis_status,
                iis_valid=bool(iis.valid_),
                validation_status="invalid_iis",
            ),
        )
    rows, bounds = _members(model, iis)
    validation = _verify(model, rows, bounds, _remaining(started, limit), relax=relax)
    metadata = _metadata(
        verified=validation == "infeasible",
        irreducible=iis_status == IIS_STATUS_IRREDUCIBLE and validation == "infeasible",
        total_rows=len(rows),
        total_bounds=len(bounds),
        rows_truncated=len(rows) > ROW_LIMIT,
        bounds_truncated=len(bounds) > BOUND_LIMIT,
        iis_status=iis_status,
        iis_valid=bool(iis.valid_),
        validation_status=validation,
        relaxation=relax,
    )
    if validation == "time_limit":
        return _report(
            model,
            started,
            "time_limit",
            _msg(
                relax,
                "IIS verification reached its time limit.",
                "LP relaxation IIS verification reached its time limit.",
            ),
            method=method,
            relaxation=relax,
            metadata=metadata,
        )
    if validation != "infeasible":
        return _report(
            model,
            started,
            "unavailable",
            _msg(
                relax,
                "HiGHS IIS failed independent verification.",
                "HiGHS LP relaxation IIS failed independent verification.",
            ),
            method=method,
            relaxation=relax,
            metadata=metadata,
        )
    edges, edges_truncated = _edges(model, rows)
    metadata = metadata.model_copy(update={"edges_truncated": edges_truncated})
    return ConflictReport(
        model_hash=model.model_hash,
        status="conflict",
        method=method,
        message=_msg(
            relax,
            "Verified IIS; irreducible is reported by HiGHS, not minimum cardinality.",
            "Verified IIS of the LP relaxation (every column continuous): these MIP rows and "
            "bounds are infeasible even without integrality, so the MIP is infeasible too. "
            "Irreducible is reported by HiGHS for the relaxation, not necessarily for the "
            "integer model, and is not minimum cardinality.",
        ),
        infeasible_rows=rows[:ROW_LIMIT],
        infeasible_bounds=bounds[:BOUND_LIMIT],
        constraint_groups=_groups(model.row_names, rows),
        variable_groups=_groups(model.column_names, bounds),
        edges=edges,
        columns=_edge_columns(model, edges),
        conflict_metadata=metadata,
        runtime_seconds=max(0.0, time.monotonic() - started),
    )


def conflict_report(model: LoadedModel, parameters: ConflictParameters) -> ConflictReport:
    """Compute and verify a HiGHS-native IIS of an infeasible LP or MIP LP relaxation."""
    started = time.monotonic()
    # A MIP's feasible set lies inside its LP relaxation's, and IIS members are a subset
    # of the MIP's own rows and bounds, so an infeasible relaxation subsystem certifies
    # that the MIP is infeasible.
    relax = _is_mip(model)
    method = RELAXATION_METHOD if relax else LP_METHOD
    # Also configure loaded instance: callers may subsequently use it directly.
    # Native workers isolate production operations, but test callers share HiGHS' scheduler.
    try:
        _require_ok(model.highs.setOptionValue("threads", 1))
    except Exception:
        return _report(
            model,
            started,
            "unavailable",
            "HiGHS conflict extraction is unavailable.",
            method=method,
        )
    if parameters.time_limit_seconds == 0:
        return _report(
            model,
            started,
            "time_limit",
            "Conflict time budget is zero.",
            method=method,
            metadata=_metadata(validation_status="time_limit"),
        )
    try:
        return _extract(model, started, parameters.time_limit_seconds, relax=relax, method=method)
    except Exception:
        return _report(
            model,
            started,
            "unavailable",
            "HiGHS conflict extraction is unavailable.",
            method=method,
            relaxation=relax,
        )
