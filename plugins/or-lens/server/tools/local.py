"""Tools intentionally available only to the local desktop stdio server."""

from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from server.core.errors import ModelError
from server.schemas.base import ErrorDetail, ErrorResult
from server.schemas.local import LocalWorkspaceOverview, OptimizationPanelState
from server.service import WorkbenchService


def register_local_tools(
    mcp: FastMCP, service: WorkbenchService, workbench_url: Callable[[str], str]
) -> None:
    """Attach local-path tools to the stdio process only.

    This function is deliberately not called by ``create_server``: adding it to
    the Streamable HTTP server would let a remote caller name local files.
    """

    def register(*, read_only: bool, idempotent: bool) -> Callable[[Callable[..., Any]], None]:
        def decorate(function: Callable[..., Any]) -> None:
            @wraps(function)
            async def guarded(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await function(*args, **kwargs)
                except ModelError as exc:
                    raise ToolError(
                        ErrorResult(
                            error=ErrorDetail(code=exc.code, message=exc.message)
                        ).model_dump_json()
                    ) from None
                except ValidationError:
                    raise ToolError(
                        '{"error":{"code":"invalid_parameters","message":"Invalid parameters."}}'
                    ) from None
                except Exception:
                    raise ToolError(
                        '{"error":{"code":"internal_error",'
                        '"message":"Could not complete operation."}}'
                    ) from None

            mcp.add_tool(
                guarded,
                annotations=ToolAnnotations(
                    readOnlyHint=read_only,
                    destructiveHint=False,
                    idempotentHint=idempotent,
                    openWorldHint=False,
                ),
            )
            tool = mcp._tool_manager.get_tool(guarded.__name__)
            assert tool is not None
            tool.fn_metadata.arg_model.model_config["hide_input_in_errors"] = True
            tool.fn_metadata.arg_model.model_rebuild(force=True)
        return decorate

    @register(read_only=False, idempotent=False)
    async def open_local_optimization_model(
        path: Annotated[str, Field(min_length=1, max_length=4096)],
    ) -> LocalWorkspaceOverview:
        """Open an absolute local .lp or .mps file in OR Lens.

        This tool copies a bounded snapshot before inspection, never changes the
        source file, and is available only in the local Codex desktop connection.
        Open ``workbench_url`` in Codex's right panel to inspect the matrix and
        diagnostics while discussing this model.
        """
        overview = await service.open_local_path(path)
        return LocalWorkspaceOverview(
            **overview.model_dump(), workbench_url=workbench_url(overview.model_id)
        )

    @register(read_only=True, idempotent=True)
    async def get_optimization_panel_state(
        model_id: Annotated[str, Field(min_length=1, max_length=128)],
    ) -> OptimizationPanelState:
        """Return the current OR Lens overview and last matrix selection for a model.

        Use the returned ``workbench_url`` to restore the existing right-side
        panel. The URL is loopback-only and remains valid until this desktop
        connection closes.
        """
        return OptimizationPanelState(
            model_id=model_id,
            workbench_url=workbench_url(model_id),
            overview=service.overview(model_id),
            selection=service.last_selection(model_id),
        )
