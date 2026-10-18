"""Pydantic schemas cho API suy luận VQA (Prediction).

Người phụ trách triển khai chính: Người 2.
Mục tiêu: Định nghĩa schema xác thực request/response cho luồng suy luận.

Hợp đồng dữ liệu thống nhất:
- Input dự kiến:
    + file: File ảnh upload (UploadFile - xử lý ở form/service)
    + question: Chuỗi câu hỏi (1 - 2000 ký tự, strip whitespace)
    + model_id: 'scratch' | 'finetuned'
- Response HTTP 200 dự kiến:
    + model_id: str
    + question: str
    + answer: str
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app.schemas.model import ModelId


# TODO (Người 2): Hoàn thiện schema request/response theo hợp đồng docs/api_contract.md
class PredictionResponse(BaseModel):
    """Schema phản hồi kết quả suy luận VQA thành công."""

    model_config = ConfigDict(extra="forbid")

    model_id: ModelId = Field(
        ...,
        description="Mã định danh model đã thực hiện suy luận",
        examples=["scratch"],
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi đã được chuẩn hóa",
        examples=["Bức ảnh này chụp danh lam thắng cảnh nào?"],
    )
    answer: str = Field(
        ...,
        min_length=1,
        description="Câu trả lời do model AI sinh ra",
        examples=["Chùa Một Cột tại Hà Nội."],
    )

    @field_validator("question", "answer")
    @classmethod
    def text_must_have_content(cls, value: str, info: ValidationInfo) -> str:
        """Chặn chuỗi chỉ gồm khoảng trắng (min_length không bắt được trường hợp này)."""
        if not value.strip():
            raise ValueError(f"{info.field_name} không được rỗng.")
        return value
