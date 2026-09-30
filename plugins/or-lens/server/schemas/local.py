"""Local-Codex-only contracts; never registered by the HTTP MCP server."""

from server.schemas.base import StrictModel
from server.schemas.matrix import SelectionContext
from server.schemas.workspace import WorkspaceOverview


class LocalWorkspaceOverview(WorkspaceOverview):
    """A local snapshot plus the loopback workbench that owns it."""

    workbench_url: str


class OptimizationPanelState(StrictModel):
    """State Codex can use to restore the panel after a follow-up question."""

    model_id: str
    workbench_url: str
    overview: WorkspaceOverview
    selection: SelectionContext | None = None
