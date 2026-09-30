from pydantic import Field

from server.schemas.base import StrictModel


class UploadedFile(StrictModel):
    """OpenAI file parameter. Optional properties stay optional in JSON Schema."""

    download_url: str = Field(min_length=1, max_length=16384)
    file_id: str = Field(min_length=1, max_length=1024)
    mime_type: str = Field(default="", max_length=256)
    file_name: str = Field(default="", max_length=512)
