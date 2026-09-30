"""Bounded views and contexts over a loaded sparse optimization matrix."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Literal

import highspy
import numpy as np
from scipy.sparse import csc_matrix

from server.schemas.matrix import (
    ColumnSelectionDetail,
    IndexedLabel,
    MatrixEntry,
    MatrixParameters,
    MatrixTile,
    MatrixView,
    MatrixWindow,
    RowSelectionDetail,
    SelectionCoefficient,
    SelectionContext,
    SelectionParameters,
)

from .errors import ModelError
from .model_loader import LoadedModel

EXACT_ENTRY_LIMIT = 2_048
TILE_LIMIT = 1_024
DETAIL_LIMIT = 10
COEFFICIENT_LIMIT = 30
LABEL_LIMIT = 50
NAME_LIMIT = 200
BASE_TILE_SIZE = 32
HIGHS_INFINITY = 1e20
BIG_M_THRESHOLD = 100_000.0


def _invalid_window(message: str) -> ModelError:
    return ModelError("invalid_matrix_window", message)


def _window(model: LoadedModel, parameters: MatrixParameters) -> MatrixWindow:
    rows, columns = model.matrix.shape
    row_end = rows if parameters.row_end is None else parameters.row_end
    column_end = columns if parameters.column_end is None else parameters.column_end
    if parameters.row_start > row_end or parameters.column_start > column_end:
        raise _invalid_window("Matrix window start must not exceed its end.")
    if row_end > rows or column_end > columns:
        raise _invalid_window("Matrix window exceeds model dimensions.")
    return MatrixWindow(
        row_start=parameters.row_start,
        row_end=row_end,
        column_start=parameters.column_start,
        column_end=column_end,
    )


def _selection_window(model: LoadedModel, parameters: SelectionParameters) -> MatrixWindow:
    window = _window(model, parameters)
    if parameters.kind == "row" and window.row_end - window.row_start != 1:
        raise _invalid_window("Row selection must contain exactly one row.")
    if parameters.kind == "column" and window.column_end - window.column_start != 1:
        raise _invalid_window("Column selection must contain exactly one column.")
    return window


def _entries(model: LoadedModel, window: MatrixWindow) -> Iterator[tuple[int, int, float]]:
    matrix = model.matrix
    for row in range(window.row_start, window.row_end):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        for column, value in zip(matrix.indices[start:end], matrix.data[start:end], strict=True):
            column_index = int(column)
            if window.column_start <= column_index < window.column_end:
                yield row, column_index, float(value)


def _bounded_name(name: str) -> tuple[str, bool]:
    return (name[:NAME_LIMIT], len(name) > NAME_LIMIT)


def _labels(names: list[str], start: int, end: int) -> tuple[list[IndexedLabel], bool]:
    shown_end = min(end, start + LABEL_LIMIT)
    labels = []
    for index in range(start, shown_end):
        name, truncated = _bounded_name(names[index])
        labels.append(IndexedLabel(index=index, name=name, name_truncated=truncated))
    return labels, end - start > LABEL_LIMIT or any(label.name_truncated for label in labels)


def _tile_size(window: MatrixWindow) -> int:
    rows = window.row_end - window.row_start
    columns = window.column_end - window.column_start
    row_tiles = math.ceil(rows / BASE_TILE_SIZE)
    column_tiles = math.ceil(columns / BASE_TILE_SIZE)
    scale = max(1, math.ceil(math.sqrt((row_tiles * column_tiles) / TILE_LIMIT)))
    size = BASE_TILE_SIZE * scale
    # Area alone underestimates tile count for thin or unevenly divided windows.
    while math.ceil(rows / size) * math.ceil(columns / size) > TILE_LIMIT:
        size += BASE_TILE_SIZE
    return size


def matrix_view(model: LoadedModel, parameters: MatrixParameters) -> MatrixView:
    """Return an exact or tiled sparse window without dense matrix expansion."""
    window = _window(model, parameters)
    visible = sum(1 for _ in _entries(model, window))
    row_labels, rows_truncated = _labels(model.row_names, window.row_start, window.row_end)
    column_labels, columns_truncated = _labels(
        model.column_names, window.column_start, window.column_end
    )
    common = dict(
        model_hash=model.model_hash,
        rows=model.matrix.shape[0],
        columns=model.matrix.shape[1],
        total_nonzeros=int(model.matrix.nnz),
        visible_nonzeros=visible,
        window=window,
        row_labels=row_labels,
        column_labels=column_labels,
        labels_truncated=rows_truncated or columns_truncated,
    )
    if visible <= EXACT_ENTRY_LIMIT:
        entries = [
            MatrixEntry(row=row, column=column, value=value)
            for row, column, value in _entries(model, window)
        ]
        return MatrixView(
            **common,
            mode="exact",
            entries=entries,
        )

    size = _tile_size(window)
    aggregates: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, math.inf, 0.0])
    for row, column, value in _entries(model, window):
        key = ((row - window.row_start) // size, (column - window.column_start) // size)
        aggregate = aggregates[key]
        magnitude = abs(value)
        aggregate[0] += 1
        aggregate[1] = min(aggregate[1], magnitude)
        aggregate[2] = max(aggregate[2], magnitude)
    tiles = []
    for (tile_row, tile_column), (nonzeros, min_abs, max_abs) in sorted(aggregates.items()):
        row_start = window.row_start + tile_row * size
        column_start = window.column_start + tile_column * size
        tiles.append(
            MatrixTile(
                row_start=row_start,
                row_end=min(row_start + size, window.row_end),
                column_start=column_start,
                column_end=min(column_start + size, window.column_end),
                nonzeros=int(nonzeros),
                min_abs=min_abs,
                max_abs=max_abs,
            )
        )
    return MatrixView(**common, mode="aggregated", tiles=tiles)


def _variable_types(model: LoadedModel) -> list[highspy.HighsVarType]:
    types = list(model.lp.integrality_)
    return types or [highspy.HighsVarType.kContinuous] * model.lp.num_col_


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) and abs(value) < HIGHS_INFINITY else None


def _domain(
    variable_type: highspy.HighsVarType, lower: float, upper: float
) -> Literal["binary", "integer", "continuous"]:
    if variable_type == highspy.HighsVarType.kInteger:
        return "binary" if lower >= 0 and upper <= 1 else "integer"
    return "continuous"


def _row_sense(
    lower: float, upper: float
) -> Literal["equality", "less_equal", "greater_equal", "ranged", "free"]:
    lower_finite, upper_finite = lower > -HIGHS_INFINITY, upper < HIGHS_INFINITY
    if lower_finite and upper_finite:
        return "equality" if lower == upper else "ranged"
    if upper_finite:
        return "less_equal"
    if lower_finite:
        return "greater_equal"
    return "free"


def _row_diagnostics(
    model: LoadedModel, row: int, binary_columns: np.ndarray
) -> list[str]:
    matrix = model.matrix
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    columns, values = matrix.indices[start:end], matrix.data[start:end]
    codes: list[str] = []
    if len(columns) == 0:
        codes.append("empty_rows")
    if len(columns) == 1:
        codes.append("singleton_rows")
    magnitudes = np.abs(values)
    nonzero_magnitudes = magnitudes[magnitudes != 0]
    if (
        nonzero_magnitudes.size
        and float(nonzero_magnitudes.max() / nonzero_magnitudes.min()) >= 1_000_000.0
    ):
        codes.append("matrix_coefficient_range")
    if any(
        binary_columns[int(column)] and abs(float(value)) >= BIG_M_THRESHOLD
        for column, value in zip(columns, values, strict=True)
    ):
        codes.append("possible_big_m")
    return codes


def _column_diagnostics(
    model: LoadedModel, csc: csc_matrix, column: int, is_binary: bool
) -> list[str]:
    """Return direct column evidence; csc is shared by one selection call."""
    column_starts = csc.indptr
    lower, upper = float(model.lp.col_lower_[column]), float(model.lp.col_upper_[column])
    codes: list[str] = []
    if column_starts[column] == column_starts[column + 1]:
        codes.append("empty_columns")
    if lower == upper:
        codes.append("fixed_variables")
    if lower <= -HIGHS_INFINITY and upper >= HIGHS_INFINITY:
        codes.append("free_variables")
    values = csc.data[column_starts[column] : column_starts[column + 1]]
    if is_binary and any(abs(float(value)) >= BIG_M_THRESHOLD for value in values):
        codes.append("possible_big_m")
    return codes


def selection_context(model: LoadedModel, parameters: SelectionParameters) -> SelectionContext:
    """Build concise, deterministic evidence for a row, column, or block selection."""
    window = _selection_window(model, parameters)
    lp, matrix = model.lp, model.matrix
    types = _variable_types(model)
    lower_columns = np.asarray(lp.col_lower_, dtype=np.float64)
    upper_columns = np.asarray(lp.col_upper_, dtype=np.float64)
    binary_columns = np.asarray(
        [
            kind == highspy.HighsVarType.kInteger and lower >= 0 and upper <= 1
            for kind, lower, upper in zip(types, lower_columns, upper_columns, strict=True)
        ],
        dtype=bool,
    )
    selected_entries = _entries(model, window)
    nonzeros = 0
    coefficients: list[SelectionCoefficient] = []
    for row, column, value in selected_entries:
        nonzeros += 1
        if len(coefficients) < COEFFICIENT_LIMIT:
            coefficients.append(SelectionCoefficient(row=row, column=column, value=value))

    row_indices = range(window.row_start, window.row_end)
    column_indices = range(window.column_start, window.column_end)
    rows: list[RowSelectionDetail] = []
    for row in row_indices:
        if len(rows) == DETAIL_LIMIT:
            break
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        values, row_columns = matrix.data[start:end], matrix.indices[start:end]
        name, name_truncated = _bounded_name(model.row_names[row])
        rows.append(
            RowSelectionDetail(
                index=row,
                name=name,
                name_truncated=name_truncated,
                lower=_finite(float(lp.row_lower_[row])),
                upper=_finite(float(lp.row_upper_[row])),
                sense=_row_sense(float(lp.row_lower_[row]), float(lp.row_upper_[row])),
                nonzeros=len(row_columns),
                largest_coefficient_magnitude=float(np.abs(values).max()) if len(values) else 0.0,
                binary_variables_involved=int(binary_columns[row_columns].sum()),
                diagnostics=_row_diagnostics(model, row, binary_columns),
            )
        )

    csc = matrix.tocsc()
    columns: list[ColumnSelectionDetail] = []
    for column in column_indices:
        if len(columns) == DETAIL_LIMIT:
            break
        name, name_truncated = _bounded_name(model.column_names[column])
        lower, upper = float(lp.col_lower_[column]), float(lp.col_upper_[column])
        columns.append(
            ColumnSelectionDetail(
                index=column,
                name=name,
                name_truncated=name_truncated,
                domain=_domain(types[column], lower, upper),
                lower=_finite(lower),
                upper=_finite(upper),
                objective=float(lp.col_cost_[column]),
                nonzeros=int(csc.indptr[column + 1] - csc.indptr[column]),
                diagnostics=_column_diagnostics(model, csc, column, bool(binary_columns[column])),
            )
        )
    row_count = window.row_end - window.row_start
    column_count = window.column_end - window.column_start
    truncated = (
        row_count > DETAIL_LIMIT
        or column_count > DETAIL_LIMIT
        or nonzeros > COEFFICIENT_LIMIT
        or any(detail.name_truncated for detail in rows)
        or any(detail.name_truncated for detail in columns)
    )
    summary = (
        f"{parameters.kind} selection: {row_count} row(s), {column_count} column(s), "
        f"{nonzeros} nonzero coefficient(s)."
    )
    return SelectionContext(
        model_hash=model.model_hash,
        kind=parameters.kind,
        window=window,
        row_count=row_count,
        column_count=column_count,
        nonzeros=nonzeros,
        rows=rows,
        columns=columns,
        coefficients=coefficients,
        truncated=truncated,
        summary=summary,
    )
