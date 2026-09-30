"""Sparse bipartite graph analysis for loaded optimization models."""

from __future__ import annotations

from dataclasses import dataclass

from server.schemas.diagnostics import ConnectedComponent, IndexedName, StructureAnalysis

from .model_loader import LoadedModel

MAX_COMPONENTS = 10
MAX_ENTITY_EXAMPLES = 20


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class _Component:
    rows: list[int]
    columns: list[int]
    nonzeros: int


def _indexed(names: list[str], indices: list[int]) -> list[IndexedName]:
    return [IndexedName(index=index, name=names[index]) for index in indices[:MAX_ENTITY_EXAMPLES]]


def component_summary(model: LoadedModel) -> StructureAnalysis:
    """Return components of matrix bipartite graph without densifying it."""
    rows, columns = model.matrix.shape
    node_count = rows + columns
    if node_count == 0:
        return StructureAnalysis(
            component_count=0,
            isolated_rows=0,
            isolated_columns=0,
            components=[],
            components_truncated=False,
        )

    union_find = _UnionFind(node_count)
    matrix = model.matrix
    for row in range(rows):
        for column in matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]:
            union_find.union(row, rows + int(column))

    groups: dict[int, _Component] = {}
    for row in range(rows):
        root = union_find.find(row)
        component = groups.get(root)
        if component is None:
            component = _Component(rows=[], columns=[], nonzeros=0)
            groups[root] = component
        component.rows.append(row)
    for column in range(columns):
        root = union_find.find(rows + column)
        component = groups.get(root)
        if component is None:
            component = _Component(rows=[], columns=[], nonzeros=0)
            groups[root] = component
        component.columns.append(column)
    for row in range(rows):
        root = union_find.find(row)
        component = groups[root]
        component.nonzeros += int(matrix.indptr[row + 1] - matrix.indptr[row])

    components = sorted(
        groups.values(),
        key=lambda item: (-(len(item.rows) + len(item.columns)), item.rows, item.columns),
    )
    isolated_rows = sum(1 for item in components if len(item.rows) == 1 and not item.columns)
    isolated_columns = sum(1 for item in components if len(item.columns) == 1 and not item.rows)
    displayed = components[:MAX_COMPONENTS]
    return StructureAnalysis(
        component_count=len(components),
        isolated_rows=isolated_rows,
        isolated_columns=isolated_columns,
        components=[
            ConnectedComponent(
                row_count=len(item.rows),
                column_count=len(item.columns),
                nonzeros=item.nonzeros,
                rows=_indexed(model.row_names, item.rows),
                columns=_indexed(model.column_names, item.columns),
                rows_truncated=len(item.rows) > MAX_ENTITY_EXAMPLES,
                columns_truncated=len(item.columns) > MAX_ENTITY_EXAMPLES,
            )
            for item in displayed
        ],
        components_truncated=len(components) > MAX_COMPONENTS,
    )
