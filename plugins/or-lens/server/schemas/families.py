"""Schemas for name-based row/column families and their exact statistics.

The grouping of rows and columns into families is a heuristic derived only from
names. Every count and range reported for a family or a family block is an
exact, deterministic fact about the members of that heuristic grouping.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel


class FamilyMember(StrictModel):
    """One concrete row or column of a family, identified by its matrix index."""

    index: int = Field(ge=0)
    name: str = Field(max_length=200)
    name_truncated: bool


class RowSenseCounts(StrictModel):
    """Exact row counts per sense, classified as in model inspection."""

    equality: int = Field(ge=0)
    less_equal: int = Field(ge=0)
    greater_equal: int = Field(ge=0)
    ranged: int = Field(ge=0)
    free: int = Field(ge=0)


class RowFamily(StrictModel):
    """Exact statistics for the rows of one heuristic name family.

    Coefficient ranges cover the family's nonzero matrix coefficients; RHS ranges
    cover its finite row bounds. Minima exclude zero; ranges are null when no
    qualifying finite value exists.
    """

    family_id: int = Field(ge=0, description="Position of this family in row_families.")
    name: str = Field(max_length=200, description="Heuristic family key derived from names.")
    name_truncated: bool
    count: int = Field(ge=1)
    senses: RowSenseCounts
    nonzeros: int = Field(ge=0)
    coefficient_min_abs: float | None = Field(ge=0.0)
    coefficient_max_abs: float | None = Field(ge=0.0)
    rhs_min_abs_nonzero: float | None = Field(ge=0.0)
    rhs_max_abs: float | None = Field(ge=0.0)
    first_index: int = Field(ge=0)
    last_index: int = Field(ge=0)
    contiguous: bool = Field(description="True when members occupy one index interval.")
    example: FamilyMember = Field(description="Member with the smallest row index.")


class ColumnFamily(StrictModel):
    """Exact statistics for the columns of one heuristic name family.

    Domains are mutually exclusive (binary, non-binary integer, continuous);
    fixed and free are independent counts. Minima exclude zero; ranges are null
    when no qualifying finite value exists.
    """

    family_id: int = Field(ge=0, description="Position of this family in column_families.")
    name: str = Field(max_length=200, description="Heuristic family key derived from names.")
    name_truncated: bool
    count: int = Field(ge=1)
    binary: int = Field(ge=0)
    integer: int = Field(ge=0, description="Integer columns that are not binary.")
    continuous: int = Field(ge=0)
    fixed: int = Field(ge=0)
    free: int = Field(ge=0)
    nonzeros: int = Field(ge=0)
    coefficient_min_abs: float | None = Field(ge=0.0)
    coefficient_max_abs: float | None = Field(ge=0.0)
    objective_min_abs_nonzero: float | None = Field(ge=0.0)
    objective_max_abs: float | None = Field(ge=0.0)
    first_index: int = Field(ge=0)
    last_index: int = Field(ge=0)
    contiguous: bool = Field(description="True when members occupy one index interval.")
    example: FamilyMember = Field(description="Member with the smallest column index.")


class FamilyBlock(StrictModel):
    """Exact nonzero count and magnitude range where two returned families meet."""

    row_family: int = Field(ge=0, description="family_id in row_families.")
    column_family: int = Field(ge=0, description="family_id in column_families.")
    nonzeros: int = Field(ge=1)
    min_abs: float = Field(ge=0.0)
    max_abs: float = Field(ge=0.0)


class ModelFamilies(StrictModel):
    """Heuristic name-based families with exact per-family and block statistics.

    Only the grouping is heuristic: names may not reflect model structure.
    Counts, domains, senses, ranges and block incidence are exact for the grouping.
    """

    model_hash: str
    classification: Literal["heuristic"] = Field(
        description="The name-based grouping is heuristic; its statistics are exact."
    )
    method: str = Field(max_length=64)
    rule: str = Field(max_length=500)
    row_family_count: int = Field(ge=0, description="Distinct row families in the model.")
    column_family_count: int = Field(ge=0, description="Distinct column families in the model.")
    row_families: list[RowFamily] = Field(max_length=200)
    column_families: list[ColumnFamily] = Field(max_length=200)
    row_families_truncated: bool
    column_families_truncated: bool
    blocks: list[FamilyBlock] = Field(max_length=2000)
    block_count: int = Field(
        ge=0, description="Distinct nonzero family blocks across all families."
    )
    blocks_truncated: bool


class FamilyParameters(StrictModel):
    """How many families per axis to return; the summary itself is always complete."""

    max_families: int = Field(default=200, ge=1, le=200)
