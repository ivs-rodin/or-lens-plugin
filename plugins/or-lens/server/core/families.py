"""Heuristic name-based row/column families with exact sparse statistics.

Rows and columns are grouped by a deterministic rule on their names only (see
``family_key``). The grouping is a heuristic: names need not reflect the model's
mathematical structure. Given a grouping, every reported count, domain, sense,
range and family-to-family block is an exact fact about the loaded HiGHS model,
computed with vectorized sparse operations and without densifying the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import highspy
import numpy as np
from scipy.sparse import csr_matrix

from server.schemas.families import (
    ColumnFamily,
    FamilyBlock,
    FamilyMember,
    ModelFamilies,
    RowFamily,
    RowSenseCounts,
)

from .model_loader import LoadedModel

FAMILY_METHOD = "name_prefix_v1"
FAMILY_RULE = (
    "Text before the first '[' or '('; otherwise the name without a trailing index "
    "suffix matching (?:[_.:-]?[0-9]+)+$. A name is its own family when the result "
    "would be empty."
)
FAMILY_LIMIT = 200
BLOCK_LIMIT = 2_000
NAME_LIMIT = 200
HIGHS_INFINITY = 1e20
_F = TypeVar("_F", RowFamily, ColumnFamily)

_DIGITS = frozenset("0123456789")
_SEPARATORS = frozenset("_.:-")


def family_key(name: str) -> str:
    """Return the heuristic family key of one row or column name (``name_prefix_v1``).

    1. A name containing ``[`` or ``(`` belongs to the text before the first such
       character: ``demand[3]`` -> ``demand``, ``x(1,2)`` -> ``x`` and Pyomo-style
       ``c_u_link(1)_`` -> ``c_u_link``. A name starting with a bracket is its own
       family: ``(x)`` -> ``(x)``.
    2. Otherwise the longest trailing index suffix matching
       ``(?:[_.:-]?[0-9]+)+$`` is removed: ``flow_1_2`` -> ``flow``, ``R0001`` ->
       ``R``, ``x12`` -> ``x``, ``a.b.3`` -> ``a.b``. A name that would become empty
       is its own family: ``123`` -> ``123``.

    Only ASCII digits are index digits, and prefixes such as Pyomo's ``c_u_`` are
    kept. The suffix is found by one backward scan instead of a regular
    expression, so untrusted names cannot cause backtracking blow-ups.
    """
    cut = name.find("[")
    round_bracket = name.find("(")
    if round_bracket >= 0 and (cut < 0 or round_bracket < cut):
        cut = round_bracket
    if cut >= 0:
        return name[:cut] if cut > 0 else name
    end = start = len(name)
    while start > 0:
        character = name[start - 1]
        # A separator belongs to the suffix only when a digit follows it.
        if character in _DIGITS or (
            character in _SEPARATORS and start < end and name[start] in _DIGITS
        ):
            start -= 1
        else:
            break
    return name[:start] if start > 0 else name


@dataclass(frozen=True)
class _Axis:
    """Families of one matrix axis; arrays are indexed by internal family id.

    Internal ids follow first appearance in model order.
    """

    keys: list[str]
    members: np.ndarray
    counts: np.ndarray
    first_index: np.ndarray
    last_index: np.ndarray


@dataclass(frozen=True)
class _Blocks:
    """Distinct nonzero (row family, column family) pairs over all families."""

    row_family: np.ndarray
    column_family: np.ndarray
    nonzeros: np.ndarray
    min_abs: np.ndarray
    max_abs: np.ndarray


@dataclass(frozen=True)
class _MatrixTotals:
    """Per-family nonzero count and magnitude range; +/-inf mark an empty family."""

    nonzeros: np.ndarray
    min_abs: np.ndarray
    max_abs: np.ndarray


def _axis(names: list[str]) -> _Axis:
    family_ids: dict[str, int] = {}
    members = np.fromiter(
        (family_ids.setdefault(family_key(name), len(family_ids)) for name in names),
        dtype=np.int64,
        count=len(names),
    )
    size = len(family_ids)
    positions = np.arange(members.size, dtype=np.int64)
    first_index = np.full(size, members.size, dtype=np.int64)
    last_index = np.full(size, -1, dtype=np.int64)
    np.minimum.at(first_index, members, positions)
    np.maximum.at(last_index, members, positions)
    return _Axis(
        keys=list(family_ids),
        members=members,
        counts=np.bincount(members, minlength=size),
        first_index=first_index,
        last_index=last_index,
    )


def _grouped_range(
    groups: np.ndarray, values: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group minimum nonzero and maximum of finite absolute values.

    Matches model inspection ranges: zero is excluded from the minimum, and a
    group without qualifying values keeps the +inf/-inf sentinel (reported null).
    """
    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(magnitudes)
    groups, magnitudes = groups[finite], magnitudes[finite]
    maximum = np.full(size, -np.inf)
    np.maximum.at(maximum, groups, magnitudes)
    nonzero = magnitudes != 0
    minimum = np.full(size, np.inf)
    np.minimum.at(minimum, groups[nonzero], magnitudes[nonzero])
    return minimum, maximum


def _matrix_blocks(matrix: csr_matrix, rows: _Axis, columns: _Axis) -> _Blocks:
    """Aggregate finite nonzero coefficients by (row family, column family).

    Each stored coefficient gets the int64 key ``row_family * column_families +
    column_family``. Distinct keys come from one sort (numpy's hash-based
    ``unique`` is much slower when most keys are distinct); every coefficient is
    then mapped to its block by binary search. Memory stays linear in nonzeros.
    """
    magnitudes = np.abs(np.asarray(matrix.data, dtype=np.float64))
    keys = np.repeat(rows.members, np.diff(matrix.indptr))
    entry_columns = columns.members[matrix.indices]
    stored = np.isfinite(magnitudes) & (magnitudes != 0)
    if not stored.all():
        magnitudes, keys, entry_columns = magnitudes[stored], keys[stored], entry_columns[stored]
    del stored
    if keys.size == 0:
        empty_ids = np.zeros(0, dtype=np.int64)
        empty_values = np.zeros(0, dtype=np.float64)
        return _Blocks(empty_ids, empty_ids, empty_ids, empty_values, empty_values)
    column_family_count = len(columns.keys)
    keys *= column_family_count
    keys += entry_columns
    del entry_columns
    sorted_keys = np.sort(keys)
    distinct = np.empty(sorted_keys.size, dtype=bool)
    distinct[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=distinct[1:])
    block_keys = sorted_keys[distinct]
    del sorted_keys, distinct
    inverse = np.searchsorted(block_keys, keys)
    del keys
    min_abs = np.full(block_keys.size, np.inf)
    max_abs = np.full(block_keys.size, -np.inf)
    np.minimum.at(min_abs, inverse, magnitudes)
    np.maximum.at(max_abs, inverse, magnitudes)
    del magnitudes
    nonzeros = np.bincount(inverse, minlength=block_keys.size)
    del inverse
    row_family = block_keys // column_family_count
    block_keys %= column_family_count  # reuse the key storage for column families
    return _Blocks(
        row_family=row_family,
        column_family=block_keys,
        nonzeros=nonzeros,
        min_abs=min_abs,
        max_abs=max_abs,
    )


def _matrix_totals(families: np.ndarray, blocks: _Blocks, size: int) -> _MatrixTotals:
    """Reduce exact block statistics to one axis's families."""
    nonzeros = np.zeros(size, dtype=np.int64)
    min_abs = np.full(size, np.inf)
    max_abs = np.full(size, -np.inf)
    np.add.at(nonzeros, families, blocks.nonzeros)
    np.minimum.at(min_abs, families, blocks.min_abs)
    np.maximum.at(max_abs, families, blocks.max_abs)
    return _MatrixTotals(nonzeros=nonzeros, min_abs=min_abs, max_abs=max_abs)


def _returned_families(axis: _Axis) -> np.ndarray:
    """Internal ids of returned families in model order.

    When there are more than FAMILY_LIMIT families, keep those with the most
    members, breaking ties by smaller first index.
    """
    ranked = np.lexsort((axis.first_index, -axis.counts))[:FAMILY_LIMIT]
    return ranked[np.argsort(axis.first_index[ranked])]


def _positions(returned: np.ndarray, size: int) -> np.ndarray:
    """Map internal family ids to returned family ids (-1 when omitted)."""
    positions = np.full(size, -1, dtype=np.int64)
    positions[returned] = np.arange(returned.size, dtype=np.int64)
    return positions


def _optional(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _bounded_name(name: str) -> tuple[str, bool]:
    return name[:NAME_LIMIT], len(name) > NAME_LIMIT


def _member(names: list[str], index: int) -> FamilyMember:
    name, truncated = _bounded_name(names[index])
    return FamilyMember(index=index, name=name, name_truncated=truncated)


def _row_families(
    model: LoadedModel, rows: _Axis, totals: _MatrixTotals, returned: np.ndarray
) -> list[RowFamily]:
    lp = model.lp
    lowers = np.asarray(lp.row_lower_, dtype=np.float64)
    uppers = np.asarray(lp.row_upper_, dtype=np.float64)
    size = len(rows.keys)
    # Same classification as model_stats._constraint_senses.
    lower_finite = lowers > -HIGHS_INFINITY
    upper_finite = uppers < HIGHS_INFINITY
    equality = lower_finite & upper_finite & (lowers == uppers)
    sense_masks = {
        "equality": equality,
        "less_equal": ~lower_finite & upper_finite,
        "greater_equal": lower_finite & ~upper_finite,
        "ranged": lower_finite & upper_finite & ~equality,
        "free": ~lower_finite & ~upper_finite,
    }
    senses = {
        sense: np.bincount(rows.members[mask], minlength=size)
        for sense, mask in sense_masks.items()
    }
    rhs_min, rhs_max = _grouped_range(
        np.concatenate((rows.members[lower_finite], rows.members[upper_finite])),
        np.concatenate((lowers[lower_finite], uppers[upper_finite])),
        size,
    )
    families: list[RowFamily] = []
    for family_id, family in enumerate(returned.tolist()):
        name, name_truncated = _bounded_name(rows.keys[family])
        count = int(rows.counts[family])
        first_index = int(rows.first_index[family])
        last_index = int(rows.last_index[family])
        families.append(
            RowFamily(
                family_id=family_id,
                name=name,
                name_truncated=name_truncated,
                count=count,
                senses=RowSenseCounts(
                    equality=int(senses["equality"][family]),
                    less_equal=int(senses["less_equal"][family]),
                    greater_equal=int(senses["greater_equal"][family]),
                    ranged=int(senses["ranged"][family]),
                    free=int(senses["free"][family]),
                ),
                nonzeros=int(totals.nonzeros[family]),
                coefficient_min_abs=_optional(totals.min_abs[family]),
                coefficient_max_abs=_optional(totals.max_abs[family]),
                rhs_min_abs_nonzero=_optional(rhs_min[family]),
                rhs_max_abs=_optional(rhs_max[family]),
                first_index=first_index,
                last_index=last_index,
                contiguous=count == last_index - first_index + 1,
                example=_member(model.row_names, first_index),
            )
        )
    return families


def _integer_columns(lp: highspy.HighsLp, column_count: int) -> np.ndarray:
    # Each property read materializes a new Python list; read it exactly once.
    integrality = lp.integrality_
    if not len(integrality):
        return np.zeros(column_count, dtype=bool)
    integer_type = highspy.HighsVarType.kInteger
    return np.fromiter(
        (kind == integer_type for kind in integrality), dtype=bool, count=column_count
    )


def _column_families(
    model: LoadedModel, columns: _Axis, totals: _MatrixTotals, returned: np.ndarray
) -> list[ColumnFamily]:
    lp = model.lp
    lowers = np.asarray(lp.col_lower_, dtype=np.float64)
    uppers = np.asarray(lp.col_upper_, dtype=np.float64)
    size = len(columns.keys)
    # Same domains as model_stats._variable_domains.
    is_integer = _integer_columns(lp, columns.members.size)
    binary = is_integer & (lowers >= 0) & (uppers <= 1)
    domain_masks = {
        "binary": binary,
        "integer": is_integer & ~binary,
        "continuous": ~is_integer,
        "fixed": lowers == uppers,
        "free": (lowers <= -HIGHS_INFINITY) & (uppers >= HIGHS_INFINITY),
    }
    domains = {
        domain: np.bincount(columns.members[mask], minlength=size)
        for domain, mask in domain_masks.items()
    }
    objective_min, objective_max = _grouped_range(
        columns.members, np.asarray(lp.col_cost_, dtype=np.float64), size
    )
    families: list[ColumnFamily] = []
    for family_id, family in enumerate(returned.tolist()):
        name, name_truncated = _bounded_name(columns.keys[family])
        count = int(columns.counts[family])
        first_index = int(columns.first_index[family])
        last_index = int(columns.last_index[family])
        families.append(
            ColumnFamily(
                family_id=family_id,
                name=name,
                name_truncated=name_truncated,
                count=count,
                binary=int(domains["binary"][family]),
                integer=int(domains["integer"][family]),
                continuous=int(domains["continuous"][family]),
                fixed=int(domains["fixed"][family]),
                free=int(domains["free"][family]),
                nonzeros=int(totals.nonzeros[family]),
                coefficient_min_abs=_optional(totals.min_abs[family]),
                coefficient_max_abs=_optional(totals.max_abs[family]),
                objective_min_abs_nonzero=_optional(objective_min[family]),
                objective_max_abs=_optional(objective_max[family]),
                first_index=first_index,
                last_index=last_index,
                contiguous=count == last_index - first_index + 1,
                example=_member(model.column_names, first_index),
            )
        )
    return families


def _returned_blocks(
    blocks: _Blocks, row_positions: np.ndarray, column_positions: np.ndarray
) -> list[FamilyBlock]:
    """Blocks between returned families: most nonzeros first, then model order.

    Ties in nonzeros are broken by returned row family, then column family.
    """
    eligible = np.flatnonzero(
        (row_positions >= 0)[blocks.row_family] & (column_positions >= 0)[blocks.column_family]
    )
    row_family = row_positions[blocks.row_family[eligible]]
    column_family = column_positions[blocks.column_family[eligible]]
    nonzeros = blocks.nonzeros[eligible]
    ranked = np.lexsort((column_family, row_family, -nonzeros))[:BLOCK_LIMIT]
    shown = ranked[np.lexsort((column_family[ranked], row_family[ranked]))]
    return [
        FamilyBlock(
            row_family=int(row_family[block]),
            column_family=int(column_family[block]),
            nonzeros=int(nonzeros[block]),
            min_abs=float(blocks.min_abs[eligible[block]]),
            max_abs=float(blocks.max_abs[eligible[block]]),
        )
        for block in shown.tolist()
    ]


def family_summary(model: LoadedModel) -> ModelFamilies:
    """Return heuristic name families with exact, bounded family and block statistics.

    At most FAMILY_LIMIT families per axis are returned (largest first, ties by
    model order, then listed in model order) and at most BLOCK_LIMIT blocks
    between returned families (most nonzeros first, then listed by family ids).
    Totals and truncation flags always describe the complete model.
    """
    rows = _axis(model.row_names)
    columns = _axis(model.column_names)
    blocks = _matrix_blocks(model.matrix, rows, columns)
    returned_rows = _returned_families(rows)
    returned_columns = _returned_families(columns)
    shown_blocks = _returned_blocks(
        blocks,
        _positions(returned_rows, len(rows.keys)),
        _positions(returned_columns, len(columns.keys)),
    )
    return ModelFamilies(
        model_hash=model.model_hash,
        classification="heuristic",
        method=FAMILY_METHOD,
        rule=FAMILY_RULE,
        row_family_count=len(rows.keys),
        column_family_count=len(columns.keys),
        row_families=_row_families(
            model,
            rows,
            _matrix_totals(blocks.row_family, blocks, len(rows.keys)),
            returned_rows,
        ),
        column_families=_column_families(
            model,
            columns,
            _matrix_totals(blocks.column_family, blocks, len(columns.keys)),
            returned_columns,
        ),
        row_families_truncated=returned_rows.size < len(rows.keys),
        column_families_truncated=returned_columns.size < len(columns.keys),
        blocks=shown_blocks,
        block_count=int(blocks.nonzeros.size),
        blocks_truncated=len(shown_blocks) < blocks.nonzeros.size,
    )


def limit_families(summary: ModelFamilies, max_families: int) -> ModelFamilies:
    """Return a smaller view of ``summary`` for context-limited callers.

    Applies the same rules as family_summary: the ``max_families`` largest
    families per axis (ties by model order) listed in model order with
    renumbered ids, then the densest blocks between kept families, at most
    ten per kept family. Totals keep describing the complete model, and the
    truncation flags compare what is shown with those totals.
    """
    block_limit = min(BLOCK_LIMIT, 10 * max_families)

    def keep(families: list[_F]) -> list[_F]:
        ranked = sorted(families, key=lambda family: (-family.count, family.first_index))
        return sorted(ranked[:max_families], key=lambda family: family.first_index)

    rows = keep(summary.row_families)
    columns = keep(summary.column_families)
    row_ids = {family.family_id: index for index, family in enumerate(rows)}
    column_ids = {family.family_id: index for index, family in enumerate(columns)}
    eligible = [
        FamilyBlock(
            row_family=row_ids[block.row_family],
            column_family=column_ids[block.column_family],
            nonzeros=block.nonzeros,
            min_abs=block.min_abs,
            max_abs=block.max_abs,
        )
        for block in summary.blocks
        if block.row_family in row_ids and block.column_family in column_ids
    ]
    ranked = sorted(
        eligible, key=lambda block: (-block.nonzeros, block.row_family, block.column_family)
    )[:block_limit]
    blocks = sorted(ranked, key=lambda block: (block.row_family, block.column_family))
    return summary.model_copy(
        update={
            "row_families": [
                family.model_copy(update={"family_id": index}) for index, family in enumerate(rows)
            ],
            "column_families": [
                family.model_copy(update={"family_id": index})
                for index, family in enumerate(columns)
            ],
            "row_families_truncated": len(rows) < summary.row_family_count,
            "column_families_truncated": len(columns) < summary.column_family_count,
            "blocks": blocks,
            "blocks_truncated": len(blocks) < summary.block_count,
        }
    )
