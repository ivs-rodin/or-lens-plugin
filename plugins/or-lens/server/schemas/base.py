from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResult(StrictModel):
    error: ErrorDetail
