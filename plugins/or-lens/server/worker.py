"""Private subprocess entry point. stdout is a two-message JSON protocol."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from server.core.errors import ModelError
from server.schemas.base import ErrorDetail, ErrorResult, StrictModel


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def main() -> None:
    # Bound BLAS threads before importing numpy/scipy in the child.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from server.core.model_loader import load_model
        from server.core.model_stats import inspect_model

        request = json.load(sys.stdin)
        model = load_model(Path(request["path"]), model_name=request["model_name"])
        emit({"stage": "loaded"})
        operation = request["operation"]
        result: StrictModel
        if operation == "inspect":
            result = inspect_model(model)
        elif operation in {"analyze", "overview"}:
            from server.core.analysis import analyze_model

            result = analyze_model(model)
        elif operation == "solve":
            from server.core.solver import solve_model
            from server.schemas.solver import SolveParameters

            result = solve_model(model, SolveParameters.model_validate(request["parameters"]))
        elif operation in {"matrix", "selection"}:
            from server.core.matrix_view import matrix_view, selection_context
            from server.schemas.matrix import MatrixParameters, SelectionParameters

            if operation == "matrix":
                result = matrix_view(model, MatrixParameters.model_validate(request["parameters"]))
            else:
                result = selection_context(
                    model, SelectionParameters.model_validate(request["parameters"])
                )
        elif operation == "conflict":
            from server.core.conflicts import conflict_report
            from server.schemas.conflicts import ConflictParameters

            result = conflict_report(
                model, ConflictParameters.model_validate(request["parameters"])
            )
        elif operation == "families":
            from server.core.families import family_summary

            result = family_summary(model)
        elif operation == "solve_validated":
            from server.core.experiments import solve_validated
            from server.schemas.experiments import ValidatedSolveParameters

            result = solve_validated(
                model, ValidatedSolveParameters.model_validate(request["parameters"])
            )
        else:
            raise ModelError("invalid_operation", "Unknown model operation.")
        emit({"result": result.model_dump(mode="json")})
    except ModelError as exc:
        emit(ErrorResult(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump())
    except Exception:
        emit(
            ErrorResult(
                error=ErrorDetail(code="worker_failed", message="Could not process the model.")
            ).model_dump()
        )


if __name__ == "__main__":
    main()
