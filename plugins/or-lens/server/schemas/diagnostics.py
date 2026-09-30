"""Schemas for deterministic model diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from .base import StrictModel
from .model import ModelInspection

DiagnosticCode = Literal[
    "matrix_coefficient_range",
    "objective_coefficient_range",
    "rhs_range",
    "possible_big_m",
    "fixed_variables",
    "free_variables",
    "empty_rows",
    "empty_columns",
    "singleton_rows",
    "disconnected_components",
]


class IndexedName(StrictModel):
    """A model entity name qualified by its zero-based matrix index."""

    index: int = Field(ge=0)
    name: str


class Diagnostic(StrictModel):
    """One bounded, deterministic fact or explicitly heuristic finding."""

    code: DiagnosticCode
    severity: Literal["info", "warning"]
    title: str
    message: str
    classification: Literal["fact", "heuristic"]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    affected_rows: list[IndexedName] = Field(default_factory=list, max_length=20)
    affected_columns: list[IndexedName] = Field(default_factory=list, max_length=20)


class ConnectedComponent(StrictModel):
    """A bounded sparse-matrix bipartite connected component."""

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    nonzeros: int = Field(ge=0)
    rows: list[IndexedName] = Field(default_factory=list, max_length=20)
    columns: list[IndexedName] = Field(default_factory=list, max_length=20)
    rows_truncated: bool = False
    columns_truncated: bool = False


class StructureAnalysis(StrictModel):
    """Sparse bipartite-graph summary, including isolated matrix nodes."""

    component_count: int = Field(ge=0)
    isolated_rows: int = Field(ge=0)
    isolated_columns: int = Field(ge=0)
    components: list[ConnectedComponent] = Field(default_factory=list, max_length=10)
    components_truncated: bool = False


class ModelAnalysis(StrictModel):
    """Inspection facts plus bounded deterministic diagnostics."""

    inspection: ModelInspection
    diagnostics: list[Diagnostic]
    structure: StructureAnalysis
