"""MCP server and local developer CLI."""

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError
from starlette.applications import Starlette

from server.core.errors import ModelError
from server.core.files import download_file, model_filename
from server.core.limits import DEFAULT_TIME_LIMIT_SECONDS, MAX_TIME_LIMIT_SECONDS
from server.core.runner import Operation, run_model
from server.schemas.base import ErrorDetail, ErrorResult
from server.schemas.diagnostics import ModelAnalysis
from server.schemas.files import UploadedFile
from server.schemas.model import ModelInspection
from server.schemas.solver import SolveParameters, SolveResult
from server.service import WorkbenchService
from server.tools.local import register_local_tools
from server.tools.workbench import workbench_tools
from server.web import app_icons, register_web


class WorkbenchServer(FastMCP):
    def __init__(
        self, service: WorkbenchService, *, close_service: bool = True, **kwargs: Any
    ) -> None:
        super().__init__("OR Lens", icons=app_icons(), **kwargs)
        self.service = service
        self.close_service = close_service

    def streamable_http_app(self) -> Starlette:
        app = super().streamable_http_app()
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: Starlette) -> AsyncIterator[None]:
            # FastMCP's tool lifespan runs per stateless request. The workspace
            # must instead live for the entire ASGI server lifecycle.
            try:
                async with original_lifespan(application):
                    yield
            finally:
                if self.close_service:
                    await self.service.close()

        app.router.lifespan_context = lifespan
        return app

    async def run_stdio_async(self) -> None:
        try:
            await super().run_stdio_async()
        finally:
            if self.close_service:
                await self.service.close()


def safe_error(exc: ModelError) -> str:
    return ErrorResult(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump_json()


def file_tool(function: Callable[..., Any], *, idempotent: bool = True) -> Tool:
    tool = Tool.from_function(
        function,
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=idempotent,
            openWorldHint=False,
        ),
        meta={"openai/fileParams": ["file"]},
    )
    # FastMCP generates an outer argument model. The nested UploadedFile config
    # alone does not prevent that model from echoing signed URLs on validation errors.
    tool.fn_metadata.arg_model.model_config["hide_input_in_errors"] = True
    tool.fn_metadata.arg_model.model_rebuild(force=True)
    return tool


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    service: WorkbenchService | None = None,
    close_service: bool = True,
    instructions: str | None = None,
) -> WorkbenchServer:
    # HTTPX INFO messages include full request URLs, including upload signatures.
    # Keep client internals quiet even when MCP configures root logging at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    public_hosts = [
        value.strip()
        for value in os.environ.get("OR_LENS_HTTP_HOSTS", "").split(",")
        if value.strip()
    ]
    allowed_hosts = [
        "127.0.0.1",
        "localhost",
        "[::1]",
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        *public_hosts,
    ]
    service = service or WorkbenchService()
    slots = service.slots

    async def execute(
        operation: Operation,
        file: UploadedFile,
        filename: str | None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            name = model_filename(file, filename)
            # Limit downloads as well as native workers. Do not build an unbounded queue.
            try:
                await asyncio.wait_for(slots.acquire(), timeout=0.1)
            except TimeoutError as exc:
                raise ModelError(
                    "server_busy", "Both model workers are busy; retry later."
                ) from exc
            try:
                with tempfile.TemporaryDirectory(prefix="or-lens-") as folder:
                    path = Path(folder) / ("model" + Path(name).suffix.lower())
                    await download_file(file, path)
                    return await run_model(operation, path, name, parameters)
            finally:
                slots.release()
        except ModelError as exc:
            raise ToolError(safe_error(exc)) from None
        except Exception:
            raise ToolError(
                safe_error(ModelError("internal_error", "Could not complete the model operation."))
            ) from None

    async def inspect_optimization_model(
        file: UploadedFile,
        filename: Annotated[str | None, Field(max_length=512)] = None,
    ) -> ModelInspection:
        """Inspect LP/MPS model structure, domains, counts and coefficient ranges.

        Use for mathematical optimization models, not spreadsheets or source code.
        Provide filename ending in .lp or .mps if the uploaded file has no file_name.
        """
        return ModelInspection.model_validate(await execute("inspect", file, filename))

    async def analyze_optimization_model(
        file: UploadedFile,
        filename: Annotated[str | None, Field(max_length=512)] = None,
    ) -> ModelAnalysis:
        """Identify LP/MIP structural and numerical issues in LP/MPS files.

        Reports coefficient ranges, fixed/free variables, empty or singleton rows,
        disconnected components, and explicitly heuristic possible Big-M candidates.
        """
        return ModelAnalysis.model_validate(await execute("analyze", file, filename))

    async def solve_optimization_model(
        file: UploadedFile,
        filename: Annotated[str | None, Field(max_length=512)] = None,
        time_limit_seconds: Annotated[
            float,
            Field(
                ge=0,
                le=MAX_TIME_LIMIT_SECONDS,
                allow_inf_nan=False,
                strict=True,
            ),
        ] = DEFAULT_TIME_LIMIT_SECONDS,
        mip_relative_gap: Annotated[
            float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)
        ] = 0.0001,
        threads: Annotated[int, Field(ge=1, le=4, strict=True)] = 1,
    ) -> SolveResult:
        """Solve an LP/MIP file with HiGHS and report deterministic solver results.

        Returns termination status, available objective/bound/gap, runtime, nodes,
        and a bounded solution sample. Time limit is in seconds, at most 300.
        """
        parameters = SolveParameters(
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
            threads=threads,
        )
        return SolveResult.model_validate(
            await execute("solve", file, filename, parameters.model_dump())
        )

    server = WorkbenchServer(
        service,
        close_service=close_service,
        host=host,
        port=port,
        tools=[
            file_tool(inspect_optimization_model),
            file_tool(analyze_optimization_model),
            file_tool(solve_optimization_model, idempotent=False),
            *workbench_tools(service),
        ],
        stateless_http=True,
        json_response=True,
        max_request_body_size=64 * 1024,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=[f"https://{name}" for name in public_hosts],
        ),
        instructions=instructions or (
            "Inspect and debug LP/MPS optimization models deterministically. "
            "Diagnostics labeled heuristic are candidates, not mathematical facts. "
            "Read solution availability and termination status before describing an objective."
            " Open the workbench with open_optimization_model to retain a temporary model ID."
            " Experiments include the baseline in their run budget; test one parameter hypothesis"
            " per trial and retain only independently validated comparable improvements."
        ),
    )
    register_web(server, service, allowed_hosts)
    return server


@dataclass
class DesktopServer:
    """One local workspace shared by a stdio MCP client and a loopback workbench."""

    mcp: WorkbenchServer
    http: WorkbenchServer
    socket: socket.socket

    @property
    def port(self) -> int:
        return int(self.socket.getsockname()[1])

    def workbench_url(self, model_id: str) -> str:
        return f"http://127.0.0.1:{self.port}/?model_id={model_id}"

    async def run(self) -> None:
        config = uvicorn.Config(
            self.http.streamable_http_app(),
            host="127.0.0.1",
            port=self.port,
            access_log=False,
            log_config=None,
        )
        http_server = uvicorn.Server(config)
        http_task = asyncio.create_task(http_server.serve(sockets=[self.socket]))
        try:
            async with asyncio.timeout(10):
                while not http_server.started:
                    if http_task.done():
                        await http_task
                    await asyncio.sleep(0.01)
            await self.mcp.run_stdio_async()
        finally:
            http_server.should_exit = True
            # stdin ending is the desktop session boundary. Cancelling service
            # work first lets run_model reap child processes before the HTTP
            # lifespan removes the temporary workspace.
            await self.mcp.service.cancel_active_operations()
            try:
                await asyncio.wait_for(asyncio.shield(http_task), timeout=5)
            except TimeoutError:
                # A stalled browser connection must not keep the desktop process
                # alive. This fallback is intentionally not used for normal EOF.
                http_server.force_exit = True
                try:
                    await asyncio.wait_for(asyncio.shield(http_task), timeout=2)
                except TimeoutError:
                    http_task.cancel()
                    await asyncio.gather(http_task, return_exceptions=True)
            finally:
                self.socket.close()
                await self.mcp.service.close()


def create_desktop_server(port: int = 0) -> DesktopServer:
    """Build the local-only desktop connection without binding a fixed port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    service = WorkbenchService()
    # The HTTP server intentionally uses the ordinary remote schema. The local
    # path opener exists only in the separate stdio server below.
    bound_port = int(listener.getsockname()[1])
    http = create_server(
        "127.0.0.1", bound_port, service=service, close_service=False
    )
    mcp = create_server(
        "127.0.0.1",
        bound_port,
        service=service,
        close_service=False,
        instructions=(
            "This is a local desktop connection. Prefer open_local_optimization_model for an "
            "absolute LP/MPS path on this computer. Its workbench_url is a loopback URL: "
            "open it in the host's side panel or browser when available, otherwise give it "
            "to the user. For follow-up questions, call "
            "get_optimization_panel_state to recover the current overview and last selection. "
            "All optimization parsing and solving stays on this computer."
        ),
    )
    desktop = DesktopServer(mcp=mcp, http=http, socket=listener)
    register_local_tools(mcp, service, desktop.workbench_url)
    return desktop


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and debug optimization models.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the MCP server")
    serve.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    desktop = commands.add_parser("desktop", help="Start local stdio MCP and loopback workbench")
    desktop.add_argument("--port", type=int, default=0)
    for operation in ("inspect", "analyze", "solve"):
        command = commands.add_parser(operation, help=f"{operation.title()} a local LP/MPS file")
        command.add_argument("path", type=Path)
        if operation == "solve":
            command.add_argument(
                "--time-limit-seconds", type=float, default=DEFAULT_TIME_LIMIT_SECONDS
            )
            command.add_argument("--mip-relative-gap", type=float, default=0.0001)
            command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "serve":
        create_server(args.host, args.port).run(transport=args.transport)
        return
    if args.command == "desktop":
        asyncio.run(create_desktop_server(args.port).run())
        return
    try:
        parameters = None
        if args.command == "solve":
            parameters = SolveParameters(
                time_limit_seconds=args.time_limit_seconds,
                mip_relative_gap=args.mip_relative_gap,
                threads=args.threads,
            ).model_dump()
        result = asyncio.run(run_model(args.command, args.path, parameters=parameters))
        print(json.dumps(result, indent=2, allow_nan=False))
    except ModelError as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except ValidationError:
        print(
            safe_error(ModelError("invalid_parameters", "Invalid solver parameters.")),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
