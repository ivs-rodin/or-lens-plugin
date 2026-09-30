"""Bounded temporary model snapshots; IDs never resolve arbitrary filesystem paths."""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from server.core.errors import ModelError
from server.core.files import MAX_FILE_BYTES, model_filename
from server.core.limits import MAX_WORKSPACE_BYTES
from server.schemas.conflicts import ConflictReport
from server.schemas.diagnostics import ModelAnalysis
from server.schemas.families import ModelFamilies
from server.schemas.files import UploadedFile
from server.schemas.workspace import ModelList, ModelSummary

MAX_MODELS = 8
MAX_TOTAL_BYTES = MAX_WORKSPACE_BYTES


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ModelEntry:
    summary: ModelSummary
    path: Path
    analysis: ModelAnalysis | None = None
    families: ModelFamilies | None = None
    # Latest conflict analysis of this immutable snapshot, whoever requested it.
    conflict: ConflictReport | None = None
    busy: int = 0


class Workspace:
    def __init__(self) -> None:
        self._temporary = TemporaryDirectory(prefix="or-lens-workspace-")
        self.root = Path(self._temporary.name)
        self.models: dict[str, ModelEntry] = {}

    def _check_capacity(self, size: int) -> None:
        if not size:
            raise ModelError("empty_file", "Model file is empty.")
        if size > MAX_FILE_BYTES:
            raise ModelError("file_too_large", "Model file exceeds 1 GiB.")
        if len(self.models) >= MAX_MODELS or (
            sum(item.summary.size_bytes for item in self.models.values()) + size > MAX_TOTAL_BYTES
        ):
            raise ModelError(
                "workspace_full", "Workspace is full. Remove a model before uploading."
            )

    def _register(self, name: str, path: Path, size: int, model_hash: str) -> ModelEntry:
        model_id = path.stem
        entry = ModelEntry(
            summary=ModelSummary(
                model_id=model_id,
                name=name,
                model_hash=model_hash,
                size_bytes=size,
                created_at=utc_now(),
            ),
            path=path,
        )
        self.models[model_id] = entry
        return entry

    def add(self, name: str, data: bytes) -> ModelEntry:
        name = model_filename(UploadedFile(download_url="local", file_id="local", file_name=name))
        self._check_capacity(len(data))
        path = self.root / (uuid4().hex + Path(name).suffix.lower())
        path.write_bytes(data)
        return self._register(name, path, len(data), sha256(data).hexdigest())

    def add_file(self, name: str, staged_path: Path) -> ModelEntry:
        """Adopt a completed private upload; ownership transfers to the workspace."""
        name = model_filename(UploadedFile(download_url="local", file_id="local", file_name=name))
        size = staged_path.stat().st_size
        self._check_capacity(size)
        with staged_path.open("rb") as source:
            model_hash = file_digest(source, "sha256").hexdigest()
        path = self.root / (uuid4().hex + Path(name).suffix.lower())
        staged_path.replace(path)
        return self._register(name, path, size, model_hash)

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self.models:
            raise ModelError("model_not_found", "Model is unavailable. Upload it again.")
        return self.models[model_id]

    def list(self) -> ModelList:
        return ModelList(models=[item.summary for item in self.models.values()])

    def delete(self, model_id: str) -> None:
        entry = self.get(model_id)
        if entry.busy:
            raise ModelError("model_busy", "Wait for this model's operation to finish.")
        entry.path.unlink(missing_ok=True)
        del self.models[model_id]

    def close(self) -> None:
        self._temporary.cleanup()
        self.models.clear()
