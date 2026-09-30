"""Same-origin development API and bundled UI. MCP remains available at /mcp."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from mcp.types import Icon
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from server.core.errors import ModelError
from server.schemas.base import ErrorDetail, ErrorResult, StrictModel
from server.schemas.conflicts import ConflictParameters
from server.schemas.families import FamilyParameters
from server.schemas.matrix import MatrixParameters, SelectionParameters
from server.schemas.workspace import (
    CompareParameters,
    RunParameters,
    StartExperimentParameters,
    TrialParameters,
)
from server.service import EXAMPLES, WorkbenchService

UI_URI = "ui://or-lens/v1/index.html"
UI_PATH = Path(__file__).parent / "static" / "index.html"
ICON_PATH = Path(__file__).parent / "static" / "icon.svg"
API_BODY_LIMIT = 64 * 1024


def app_icons() -> list[Icon]:
    """The OR Lens mark as an embedded SVG, so hosts need no extra request."""
    if not ICON_PATH.is_file():
        return []
    data = base64.b64encode(ICON_PATH.read_bytes()).decode("ascii")
    return [Icon(src=f"data:image/svg+xml;base64,{data}", mimeType="image/svg+xml", sizes=["any"])]


def bundled_html() -> str:
    if not UI_PATH.is_file():
        return (
            "<!doctype html><title>OR Lens</title><h1>OR Lens</h1>"
            "<p>Build the interface: "
            "<code>npm ci &amp;&amp; npm run build:ui</code>.</p>"
            '<p><a href="/docs">Local testing instructions</a></p>'
        )
    return UI_PATH.read_text()


def json_response(result: StrictModel) -> JSONResponse:
    return JSONResponse(result.model_dump(mode="json"))


async def body_json(request: Request) -> dict:
    data = bytearray()
    async with asyncio.timeout(10):
        async for chunk in request.stream():
            if len(data) + len(chunk) > API_BODY_LIMIT:
                raise ModelError("request_too_large", "JSON request exceeds 64 KiB.")
            data.extend(chunk)
    try:
        result = json.loads(data) if data else {}
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModelError("invalid_request", "Request must contain valid JSON.") from exc
    if not isinstance(result, dict):
        raise ModelError("invalid_request", "Request must contain a JSON object.")
    return result


def register_web(mcp: FastMCP, service: WorkbenchService, allowed_hosts: list[str]) -> None:
    def route(
        path: str, methods: list[str]
    ) -> Callable[
        [Callable[[Request], Awaitable[Response]]], Callable[[Request], Awaitable[Response]]
    ]:
        def decorate(
            handler: Callable[[Request], Awaitable[Response]],
        ) -> Callable[[Request], Awaitable[Response]]:
            async def guarded(request: Request) -> Response:
                try:
                    host = request.headers.get("host", "").lower()
                    base, separator, port = host.rpartition(":")
                    hostname = base if separator and port.isdecimal() else host
                    if host not in allowed_hosts and f"{hostname}:*" not in allowed_hosts:
                        return JSONResponse(
                            {"error": {"code": "invalid_host", "message": "Invalid Host header."}},
                            status_code=421,
                        )
                    if request.method in {"POST", "DELETE"}:
                        if request.headers.get("x-or-lens") != "1":
                            raise ModelError(
                                "invalid_request", "This API requires the X-OR-Lens header."
                            )
                        origin = request.headers.get("origin")
                        if origin and (
                            urlsplit(origin).netloc.lower() != host
                            or urlsplit(origin).scheme != request.url.scheme
                        ):
                            raise ModelError("invalid_origin", "Use this API from its own origin.")
                    return await handler(request)
                except ModelError as exc:
                    status = 404 if exc.code.endswith("not_found") else 400
                    if exc.code in {"server_busy", "model_busy", "experiment_busy"}:
                        status = 409
                    if exc.code in {"file_too_large", "request_too_large"}:
                        status = 413
                    return JSONResponse(
                        ErrorResult(
                            error=ErrorDetail(code=exc.code, message=exc.message)
                        ).model_dump(),
                        status_code=status,
                    )
                except ValidationError:
                    return JSONResponse(
                        {
                            "error": {
                                "code": "invalid_parameters",
                                "message": "Invalid request parameters.",
                            }
                        },
                        status_code=400,
                    )
                except TimeoutError:
                    return JSONResponse(
                        {"error": {"code": "request_timeout", "message": "Request timed out."}},
                        status_code=408,
                    )
                except Exception:
                    return JSONResponse(
                        {
                            "error": {
                                "code": "internal_error",
                                "message": "Could not complete this operation.",
                            }
                        },
                        status_code=500,
                    )

            mcp.custom_route(path, methods=methods)(guarded)
            return guarded

        return decorate

    @route("/", ["GET"])
    async def index(request: Request) -> Response:
        return HTMLResponse(
            bundled_html(),
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "img-src data:; connect-src 'self'; base-uri 'none'; "
                    "form-action 'self'; frame-ancestors 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @route("/docs", ["GET"])
    async def docs(request: Request) -> Response:
        return HTMLResponse("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OR Lens · Local testing</title>
<style>body{font:16px system-ui;max-width:780px;margin:60px auto;padding:0 24px;
color:#172a31;background:#f7f8f5}a{color:#086c69}
pre{padding:20px;background:#e9efec;overflow:auto}li{margin:12px 0}</style>
<a href="/">← Open workbench</a><h1>Test a real optimization model</h1>
<ol><li>Open the workbench and load <b>Small LP</b>, or upload a .lp/.mps file.</li>
<li>Inspect the overview, then solve. The Small LP objective should be <b>7</b>.</li>
<li>Open Matrix to select a row, column, or block. In an MCP host the selection
becomes model context; locally use Copy context.</li>
<li>Open Families to see rows and columns grouped by name prefix (a heuristic
grouping with exact statistics) and jump from a family into the matrix.</li>
<li>Load Infeasible LP and open Conflicts to extract and verify a conflict. For a
MIP, the conflict comes from its LP relaxation when that relaxation is infeasible.</li>
<li>Open Experiments. Start a budget including the baseline, optionally with several
seeds per run, then test one parameter hypothesis at a time.</li></ol>
<h2>Terminal alternative</h2><pre>uv run or-lens inspect tests/fixtures/tiny.lp
uv run or-lens solve tests/fixtures/tiny.lp
uv run pytest</pre><h2>MCP</h2>
<p>The MCP endpoint is <code>/mcp</code>. It expects an MCP client; opening it as a
normal webpage is not a protocol test.</p>
<p>Remote MCP hosts such as ChatGPT need a reachable HTTPS endpoint or a supported
tunnel. Refresh the connection after changing tools. State is temporary and
cleared when this server stops.</p>
</html>""")

    @route("/favicon.ico", ["GET"])
    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

    @route("/api/models", ["GET"])
    async def models(request: Request) -> Response:
        return json_response(service.workspace.list())

    @route("/api/models", ["POST"])
    async def upload(request: Request) -> Response:
        body_read = asyncio.Event()

        async def chunks() -> AsyncIterator[bytes]:
            async for chunk in request.stream():
                yield chunk
            body_read.set()

        async def disconnected() -> None:
            # Only one consumer may read ASGI events while the body is arriving.
            await body_read.wait()
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    return

        processing = asyncio.create_task(
            service.open_stream(request.query_params.get("filename", ""), chunks())
        )
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                (processing, disconnect), return_when=asyncio.FIRST_COMPLETED
            )
            if processing in done:
                return json_response(await processing)
            return Response(status_code=499)
        finally:
            # Cancellation propagates to run_model, which kills/reaps its worker;
            # open_stream then removes the unfinished workspace snapshot.
            for task in (processing, disconnect):
                if not task.done():
                    task.cancel()
            await asyncio.gather(processing, disconnect, return_exceptions=True)

    @route("/api/examples", ["GET"])
    async def examples(request: Request) -> Response:
        return json_response(EXAMPLES)

    @route("/api/examples/{name}", ["POST"])
    async def example(request: Request) -> Response:
        return json_response(await service.open_example(request.path_params["name"]))

    @route("/api/models/{model_id}", ["GET"])
    async def overview(request: Request) -> Response:
        return json_response(service.overview(request.path_params["model_id"]))

    @route("/api/models/{model_id}", ["DELETE"])
    async def delete(request: Request) -> Response:
        return json_response(service.delete(request.path_params["model_id"]))

    @route("/api/models/{model_id}/matrix", ["GET"])
    async def matrix(request: Request) -> Response:
        return json_response(
            await service.matrix(
                request.path_params["model_id"],
                MatrixParameters.model_validate(dict(request.query_params)),
            )
        )

    @route("/api/models/{model_id}/families", ["GET"])
    async def families(request: Request) -> Response:
        return json_response(
            await service.families(
                request.path_params["model_id"],
                FamilyParameters.model_validate(dict(request.query_params)),
            )
        )

    @route("/api/models/{model_id}/selection", ["POST"])
    async def selection(request: Request) -> Response:
        return json_response(
            await service.selection(
                request.path_params["model_id"],
                SelectionParameters.model_validate(await body_json(request)),
            )
        )

    @route("/api/models/{model_id}/solve", ["POST"])
    async def solve(request: Request) -> Response:
        return json_response(
            await service.solve(
                request.path_params["model_id"],
                RunParameters.model_validate(await body_json(request)),
            )
        )

    @route("/api/models/{model_id}/conflict", ["POST"])
    async def conflict(request: Request) -> Response:
        return json_response(
            await service.conflict(
                request.path_params["model_id"],
                ConflictParameters.model_validate(await body_json(request)),
            )
        )

    @route("/api/models/{model_id}/runs", ["GET"])
    async def runs(request: Request) -> Response:
        return json_response(service.list_runs(request.path_params["model_id"]))

    @route("/api/compare", ["POST"])
    async def compare(request: Request) -> Response:
        params = CompareParameters.model_validate(await body_json(request))
        return json_response(service.compare(params.baseline_run_id, params.candidate_run_id))

    @route("/api/experiments", ["POST"])
    async def start(request: Request) -> Response:
        return json_response(
            await service.start_experiment(
                StartExperimentParameters.model_validate(await body_json(request))
            )
        )

    @route("/api/experiments/{experiment_id}", ["GET"])
    async def experiment(request: Request) -> Response:
        return json_response(service.session(request.path_params["experiment_id"]))

    @route("/api/experiments/{experiment_id}/trial", ["POST"])
    async def trial(request: Request) -> Response:
        return json_response(
            await service.trial(
                request.path_params["experiment_id"],
                TrialParameters.model_validate(await body_json(request)),
            )
        )

    @route("/api/experiments/{experiment_id}/cancel", ["POST"])
    async def cancel(request: Request) -> Response:
        return json_response(await service.cancel(request.path_params["experiment_id"]))

    @mcp.resource(
        UI_URI,
        name="OR Lens workbench",
        mime_type="text/html;profile=mcp-app",
        icons=app_icons(),
        meta={"ui": {"prefersBorder": True, "csp": {"connectDomains": [], "resourceDomains": []}}},
    )
    def app_resource() -> str:
        return bundled_html()
