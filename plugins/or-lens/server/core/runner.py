"""Process isolation for native parsing and solving, shared by MCP and CLI."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

from server.core.errors import ModelError
from server.core.limits import (
    DEFAULT_TIME_LIMIT_SECONDS,
    TRANSFER_TIMEOUT_SECONDS,
)

Operation = Literal[
    "inspect",
    "analyze",
    "solve",
    "overview",
    "matrix",
    "selection",
    "conflict",
    "solve_validated",
    "families",
]
PARSE_TIMEOUT_SECONDS = TRANSFER_TIMEOUT_SECONDS
ANALYSIS_TIMEOUT_SECONDS = TRANSFER_TIMEOUT_SECONDS
MAX_OUTPUT_BYTES = 1024 * 1024


async def run_model(
    operation: Operation,
    path: Path,
    model_name: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = json.dumps(
        {
            "operation": operation,
            "path": str(path.resolve()),
            "model_name": model_name or path.name,
            "parameters": parameters or {},
        },
        allow_nan=False,
    ).encode()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "server.worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=MAX_OUTPUT_BYTES,
    )
    assert process.stdin is not None and process.stdout is not None
    stage = "parse"
    try:
        process.stdin.write(request)
        await process.stdin.drain()
        process.stdin.close()
        first = await asyncio.wait_for(process.stdout.readline(), PARSE_TIMEOUT_SECONDS)
        if not first:
            raise ModelError("worker_failed", "Model worker ended without a result.")
        response = json.loads(first)
        if response.get("stage") == "loaded":
            stage = "operation"
            if operation in {"solve", "conflict"}:
                timeout = (
                    float((parameters or {}).get("time_limit_seconds", DEFAULT_TIME_LIMIT_SECONDS))
                    + 5
                )
            elif operation == "solve_validated":
                timeout = (
                    float(
                        (parameters or {})
                        .get("solve", {})
                        .get("time_limit_seconds", DEFAULT_TIME_LIMIT_SECONDS)
                    )
                    + 5
                )
            else:
                timeout = ANALYSIS_TIMEOUT_SECONDS
            line = await asyncio.wait_for(process.stdout.readline(), timeout)
            if not line:
                raise ModelError("worker_failed", "Model worker ended without a result.")
            response = json.loads(line)
        await asyncio.wait_for(process.wait(), 2)
        if "error" in response:
            raise ModelError(response["error"]["code"], response["error"]["message"])
        if process.returncode != 0 or not isinstance(response.get("result"), dict):
            raise ModelError("worker_failed", "Model worker could not complete the operation.")
        return dict(response["result"])
    except TimeoutError as exc:
        code = "parse_timeout" if stage == "parse" else "operation_timeout"
        raise ModelError(code, f"Model {stage} exceeded its wall-clock limit.") from exc
    except (ValueError, KeyError, BrokenPipeError) as exc:
        raise ModelError("worker_failed", "Model worker returned an invalid response.") from exc
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
