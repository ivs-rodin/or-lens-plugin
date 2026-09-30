"""Bounded sparse-matrix and selection schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel


class MatrixParameters(StrictModel):
    row_start: int = Field(default=0, ge=0)
    row_end: int | None = Field(default=None, ge=0)
    column_start: int = Field(default=0, ge=0)
    column_end: int | None = Field(default=None, ge=0)


class SelectionParameters(MatrixParameters):
    kind: Literal["row", "column", "block"]


class MatrixWindow(StrictModel):
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=0)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=0)


class MatrixEntry(StrictModel):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    value: float


class MatrixTile(StrictModel):
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=0)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=0)
    nonzeros: int = Field(ge=0)
    min_abs: float = Field(ge=0.0)
    max_abs: float = Field(ge=0.0)


class IndexedLabel(StrictModel):
    index: int = Field(ge=0)
    name: str = Field(max_length=200)
    name_truncated: bool = False


class MatrixView(StrictModel):
    model_hash: str
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    total_nonzeros: int = Field(ge=0)
    visible_nonzeros: int = Field(ge=0)
    window: MatrixWindow
    mode: Literal["exact", "aggregated"]
    entries: list[MatrixEntry] = Field(default_factory=list, max_length=2048)
    tiles: list[MatrixTile] = Field(default_factory=list, max_length=1024)
    row_labels: list[IndexedLabel] = Field(default_factory=list, max_length=50)
    column_labels: list[IndexedLabel] = Field(default_factory=list, max_length=50)
    labels_truncated: bool = False


class RowSelectionDetail(StrictModel):
    index: int = Field(ge=0)
    name: str = Field(max_length=200)
    name_truncated: bool = False
    lower: float | None = None
    upper: float | None = None
    sense: Literal["equality", "less_equal", "greater_equal", "ranged", "free"]
    nonzeros: int = Field(ge=0)
    largest_coefficient_magnitude: float = Field(ge=0.0)
    binary_variables_involved: int = Field(ge=0)
    diagnostics: list[str] = Field(default_factory=list)


class ColumnSelectionDetail(StrictModel):
    index: int = Field(ge=0)
    name: str = Field(max_length=200)
    name_truncated: bool = False
    domain: Literal["binary", "integer", "continuous"]
    lower: float | None = None
    upper: float | None = None
    objective: float
    nonzeros: int = Field(ge=0)
    diagnostics: list[str] = Field(default_factory=list)


class SelectionCoefficient(StrictModel):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    value: float


class SelectionContext(StrictModel):
    model_hash: str
    kind: Literal["row", "column", "block"]
    window: MatrixWindow
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    nonzeros: int = Field(ge=0)
    rows: list[RowSelectionDetail] = Field(default_factory=list, max_length=10)
    columns: list[ColumnSelectionDetail] = Field(default_factory=list, max_length=10)
    coefficients: list[SelectionCoefficient] = Field(default_factory=list, max_length=30)
    truncated: bool = False
    summary: str = Field(max_length=500)
