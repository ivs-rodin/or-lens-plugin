"""Deterministic numerical and structural diagnostics for loaded models."""

from __future__ import annotations

import math
from typing import cast

import highspy
import numpy as np
from pydantic import JsonValue

from server.schemas.diagnostics import Diagnostic, DiagnosticCode, IndexedName, ModelAnalysis
from server.schemas.model import CoefficientRange

from .graph_analysis import component_summary
from .model_loader import LoadedModel
from .model_stats import inspect_model

RANGE_RATIO_THRESHOLD = 1_000_000.0
POSSIBLE_BIG_M_THRESHOLD = 100_000.0
MAX_AFFECTED_ENTITIES = 20
MAX_BIG_M_EXAMPLES = 50


def _references(names: list[str], indices: np.ndarray) -> list[IndexedName]:
    return [
        IndexedName(index=int(index), name=names[int(index)])
        for index in indices[:MAX_AFFECTED_ENTITIES]
    ]


def _range_ratio(value_range: CoefficientRange) -> tuple[float | None, bool] | None:
    if value_range.min_abs_nonzero is None or value_range.max_abs is None:
        return None
    log_ratio = math.log10(value_range.max_abs) - math.log10(value_range.min_abs_nonzero)
    if log_ratio < math.log10(RANGE_RATIO_THRESHOLD):
        return None
    ratio = value_range.max_abs / value_range.min_abs_nonzero
    return (ratio if math.isfinite(ratio) else None, not math.isfinite(ratio))


def _range_diagnostic(
    code: DiagnosticCode, title: str, value_range: CoefficientRange
) -> Diagnostic | None:
    ratio_result = _range_ratio(value_range)
    if ratio_result is None:
        return None
    ratio, ratio_overflow = ratio_result
    ratio_text = f"{ratio:g}" if ratio is not None else "beyond finite JSON representation"
    return Diagnostic(
        code=code,
        severity="warning",
        title=title,
        message=f"Finite absolute values span a ratio of {ratio_text}; inspect numerical scaling.",
        classification="fact",
        metadata={
            "min_abs_nonzero": value_range.min_abs_nonzero,
            "max_abs": value_range.max_abs,
            "ratio": ratio,
            "ratio_overflow": ratio_overflow,
            "threshold": RANGE_RATIO_THRESHOLD,
        },
    )


def _variable_types(model: LoadedModel) -> list[highspy.HighsVarType]:
    integrality = list(model.lp.integrality_)
    if not integrality:
        return [highspy.HighsVarType.kContinuous] * model.lp.num_col_
    return integrality


def _possible_big_m(model: LoadedModel) -> Diagnostic | None:
    matrix = model.matrix
    lowers = np.asarray(model.lp.col_lower_, dtype=np.float64)
    uppers = np.asarray(model.lp.col_upper_, dtype=np.float64)
    variable_types = _variable_types(model)
    binary_columns = np.asarray(
        [
            variable_type == highspy.HighsVarType.kInteger and lower >= 0 and upper <= 1
            for variable_type, lower, upper in zip(variable_types, lowers, uppers, strict=True)
        ],
        dtype=bool,
    )
    examples: list[JsonValue] = []
    affected_rows = np.zeros(matrix.shape[0], dtype=bool)
    affected_columns = np.zeros(matrix.shape[1], dtype=bool)
    candidate_count = 0
    for row in range(matrix.shape[0]):
        for column, coefficient in zip(
            matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]],
            matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]],
            strict=True,
        ):
            column_index = int(column)
            if (
                not binary_columns[column_index]
                or abs(float(coefficient)) < POSSIBLE_BIG_M_THRESHOLD
            ):
                continue
            candidate_count += 1
            affected_rows[row] = True
            affected_columns[column_index] = True
            if len(examples) < MAX_BIG_M_EXAMPLES:
                row_reference: dict[str, JsonValue] = {
                    "index": row,
                    "name": model.row_names[row],
                }
                column_reference: dict[str, JsonValue] = {
                    "index": column_index,
                    "name": model.column_names[column_index],
                }
                examples.append(
                    {
                        "row": row_reference,
                        "column": column_reference,
                        "coefficient": float(coefficient),
                    }
                )
    if candidate_count == 0:
        return None
    return Diagnostic(
        code="possible_big_m",
        severity="warning",
        title="Possible Big-M coefficients",
        message=(
            f"{candidate_count} large coefficient(s) occur on binary columns; "
            "these are candidates, not a proof of Big-M modeling."
        ),
        classification="heuristic",
        metadata={
            "candidate_count": candidate_count,
            "affected_row_count": int(affected_rows.sum()),
            "affected_column_count": int(affected_columns.sum()),
            "examples": examples,
            "examples_truncated": candidate_count > MAX_BIG_M_EXAMPLES,
            "threshold": POSSIBLE_BIG_M_THRESHOLD,
        },
        affected_rows=_references(model.row_names, np.flatnonzero(affected_rows)),
        affected_columns=_references(model.column_names, np.flatnonzero(affected_columns)),
    )


def _structural_diagnostic(
    code: DiagnosticCode,
    title: str,
    message: str,
    count: int,
    names: list[str],
    indices: np.ndarray,
    entity: str,
) -> Diagnostic | None:
    if count == 0:
        return None
    rows = _references(names, indices) if entity == "row" else []
    columns = _references(names, indices) if entity == "column" else []
    return Diagnostic(
        code=code,
        severity="info",
        title=title,
        message=message,
        classification="fact",
        metadata={"count": count, "examples_truncated": count > MAX_AFFECTED_ENTITIES},
        affected_rows=rows,
        affected_columns=columns,
    )


def analyze_model(model: LoadedModel) -> ModelAnalysis:
    """Return inspection facts, bounded diagnostics, and sparse components."""
    inspection = inspect_model(model)
    matrix = model.matrix
    row_counts = np.diff(matrix.indptr)
    column_counts = np.diff(matrix.tocsc().indptr)
    diagnostics: list[Diagnostic] = []
    for code, title, value_range in (
        (
            "matrix_coefficient_range",
            "Wide matrix coefficient range",
            CoefficientRange(
                min_abs_nonzero=inspection.coefficient_ranges.matrix_min_abs_nonzero,
                max_abs=inspection.coefficient_ranges.matrix_max_abs,
            ),
        ),
        (
            "objective_coefficient_range",
            "Wide objective coefficient range",
            CoefficientRange(
                min_abs_nonzero=inspection.coefficient_ranges.objective_min_abs_nonzero,
                max_abs=inspection.coefficient_ranges.objective_max_abs,
            ),
        ),
        (
            "rhs_range",
            "Wide right-hand-side range",
            CoefficientRange(
                min_abs_nonzero=inspection.coefficient_ranges.rhs_min_abs_nonzero,
                max_abs=inspection.coefficient_ranges.rhs_max_abs,
            ),
        ),
    ):
        diagnostic = _range_diagnostic(cast(DiagnosticCode, code), title, value_range)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    big_m = _possible_big_m(model)
    if big_m is not None:
        diagnostics.append(big_m)
    lowers = np.asarray(model.lp.col_lower_, dtype=np.float64)
    uppers = np.asarray(model.lp.col_upper_, dtype=np.float64)
    structural = (
        (
            "fixed_variables",
            "Fixed variables",
            "Variables have equal finite lower and upper bounds.",
            np.flatnonzero(lowers == uppers),
            model.column_names,
            "column",
        ),
        (
            "free_variables",
            "Free variables",
            "Variables have no finite lower or upper bound.",
            np.flatnonzero((lowers <= -1e20) & (uppers >= 1e20)),
            model.column_names,
            "column",
        ),
        (
            "empty_rows",
            "Empty rows",
            "Constraints have no matrix coefficients.",
            np.flatnonzero(row_counts == 0),
            model.row_names,
            "row",
        ),
        (
            "empty_columns",
            "Empty columns",
            "Variables have no matrix coefficients.",
            np.flatnonzero(column_counts == 0),
            model.column_names,
            "column",
        ),
        (
            "singleton_rows",
            "Singleton rows",
            "Constraints have exactly one matrix coefficient.",
            np.flatnonzero(row_counts == 1),
            model.row_names,
            "row",
        ),
    )
    for code, title, message, indices, names, entity in structural:
        diagnostic = _structural_diagnostic(
            cast(DiagnosticCode, code), title, message, int(indices.size), names, indices, entity
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    structure = component_summary(model)
    if structure.component_count > 1:
        diagnostics.append(
            Diagnostic(
                code="disconnected_components",
                severity="info",
                title="Disconnected matrix components",
                message=(
                    f"Matrix bipartite graph has {structure.component_count} "
                    "disconnected components; "
                    "this does not itself establish a decomposable optimization problem."
                ),
                classification="fact",
                metadata={
                    "component_count": structure.component_count,
                    "components_shown": len(structure.components),
                    "components_truncated": structure.components_truncated,
                },
            )
        )
    return ModelAnalysis(inspection=inspection, diagnostics=diagnostics, structure=structure)
