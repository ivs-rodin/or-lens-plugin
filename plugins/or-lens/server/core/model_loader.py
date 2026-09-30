"""Load untrusted LP/MPS files through HiGHS into a sparse representation."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import highspy
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from .errors import ModelError
from .limits import MAX_FILE_BYTES

MAX_COLUMNS = 1_000_000
MAX_ROWS = 1_000_000
MAX_NONZEROS = 10_000_000
SUPPORTED_SUFFIXES = frozenset({".lp", ".mps"})


@dataclass(frozen=True)
class LoadedModel:
    highs: highspy.Highs
    lp: highspy.HighsLp
    matrix: csr_matrix
    model_name: str
    model_hash: str
    row_names: list[str]
    column_names: list[str]


def _model_error(code: str, message: str) -> ModelError:
    return ModelError(code=code, message=message)


def _stage_file(path: Path, destination: Path) -> str:
    """Copy and hash a stable parser snapshot without buffering the whole file."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise _model_error("unsupported_format", "Model file must have .lp or .mps extension.")
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise _model_error("model_unreadable", "Model file cannot be read.") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise _model_error("model_not_regular_file", "Model path must be a regular file.")
    if file_stat.st_size == 0:
        raise _model_error("empty_model_file", "Model file is empty.")
    if file_stat.st_size > MAX_FILE_BYTES:
        raise _model_error("file_too_large", "Model file exceeds 1 GiB limit.")
    digest = sha256()
    size = 0
    try:
        with path.open("rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise _model_error("model_not_regular_file", "Model path must be a regular file.")
            with destination.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise _model_error("file_too_large", "Model file exceeds 1 GiB limit.")
                    digest.update(chunk)
                    output.write(chunk)
    except OSError as exc:
        raise _model_error("model_unreadable", "Model file cannot be read.") from exc
    if not size:
        raise _model_error("empty_model_file", "Model file is empty.")
    return digest.hexdigest()


def _make_matrix(lp: highspy.HighsLp) -> csr_matrix:
    matrix = lp.a_matrix_
    starts = np.asarray(matrix.start_, dtype=np.int64)
    indices = np.asarray(matrix.index_, dtype=np.int64)
    values = np.asarray(matrix.value_, dtype=np.float64)
    if matrix.format_ == highspy.MatrixFormat.kColwise:
        result = csc_matrix((values, indices, starts), shape=(lp.num_row_, lp.num_col_)).tocsr()
    elif matrix.format_ == highspy.MatrixFormat.kRowwise:
        result = csr_matrix((values, indices, starts), shape=(lp.num_row_, lp.num_col_))
    else:
        raise _model_error("unsupported_matrix_format", "Model matrix format is unsupported.")
    result.eliminate_zeros()
    return result


def _reject_unsupported_types(lp: highspy.HighsLp) -> None:
    if not lp.integrality_:
        return
    unsupported = {
        highspy.HighsVarType.kSemiContinuous,
        highspy.HighsVarType.kSemiInteger,
    }
    if any(variable_type in unsupported for variable_type in lp.integrality_):
        raise _model_error(
            "unsupported_variable_domain",
            "Semi-continuous and semi-integer variables are unsupported.",
        )


def _reject_quadratic(highs: highspy.Highs) -> None:
    model = highs.getModel()
    hessian = model.hessian_
    if hessian.dim_ != 0 or len(hessian.value_) != 0:
        raise _model_error("unsupported_quadratic", "Quadratic objectives are unsupported.")


def _validate_limits(lp: highspy.HighsLp, matrix: csr_matrix) -> None:
    if lp.num_col_ > MAX_COLUMNS or lp.num_row_ > MAX_ROWS:
        raise _model_error("model_too_large", "Model dimensions exceed allowed limit.")
    if matrix.nnz > MAX_NONZEROS:
        raise _model_error("model_too_large", "Model nonzero count exceeds allowed limit.")


def _validate_numeric_data(lp: highspy.HighsLp) -> None:
    finite_values = np.concatenate(
        (
            np.asarray(lp.col_cost_, dtype=np.float64),
            np.asarray(lp.a_matrix_.value_, dtype=np.float64),
            np.asarray([lp.offset_], dtype=np.float64),
        )
    )
    bounds = np.concatenate(
        (
            np.asarray(lp.col_lower_, dtype=np.float64),
            np.asarray(lp.col_upper_, dtype=np.float64),
            np.asarray(lp.row_lower_, dtype=np.float64),
            np.asarray(lp.row_upper_, dtype=np.float64),
        )
    )
    if not np.isfinite(finite_values).all() or np.isnan(bounds).any():
        raise _model_error("nonfinite_model_data", "Model contains non-finite numeric data.")


def _model_names(lp: highspy.HighsLp) -> tuple[list[str], list[str]]:
    """Copy HiGHS name vectors once before assigning deterministic fallbacks."""
    source_row_names = lp.row_names_
    source_column_names = lp.col_names_
    row_names = [
        source_row_names[index]
        if index < len(source_row_names) and source_row_names[index]
        else f"row_{index}"
        for index in range(lp.num_row_)
    ]
    column_names = [
        source_column_names[index]
        if index < len(source_column_names) and source_column_names[index]
        else f"column_{index}"
        for index in range(lp.num_col_)
    ]
    return row_names, column_names


def load_model(path: Path, model_name: str | None = None) -> LoadedModel:
    """Load one LP/MPS file, rejecting parser warnings and unsupported models."""
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("log_to_console", False)
    # Lowest value accepted by HiGHS 1.15, reducing dropped-coefficient loss.
    highs.setOptionValue("small_matrix_value", 1e-12)
    with TemporaryDirectory(prefix="or-lens-model-") as temporary_directory:
        staged_path = Path(temporary_directory) / f"model{path.suffix.lower()}"
        model_hash = _stage_file(path, staged_path)
        try:
            status = highs.readModel(str(staged_path))
        except Exception as exc:  # highspy reports some parse errors as exceptions.
            raise _model_error("model_parse_error", "HiGHS could not parse model file.") from exc
    if status == highspy.HighsStatus.kError:
        raise _model_error("model_parse_error", "HiGHS could not parse model file.")
    lp = highs.getLp()
    try:
        # Prefer an explicit unsupported-domain result when HiGHS reports that
        # domain while also warning about its input representation.
        _reject_quadratic(highs)
        _reject_unsupported_types(lp)
        _validate_numeric_data(lp)
        if status == highspy.HighsStatus.kWarning:
            raise _model_error("model_parse_warning", "HiGHS modified model file during parsing.")
        matrix = _make_matrix(lp)
        _validate_limits(lp, matrix)
    except ModelError:
        raise
    except Exception as exc:
        raise _model_error("model_parse_error", "Model representation is invalid.") from exc
    resolved_name = model_name or path.stem
    row_names, column_names = _model_names(lp)
    return LoadedModel(
        highs=highs,
        lp=lp,
        matrix=matrix,
        model_name=resolved_name,
        model_hash=model_hash,
        row_names=row_names,
        column_names=column_names,
    )
