"""Pydantic schemas cho phản hồi lỗi chuẩn (Error Response).

Người phụ trách triển khai chính: Người 2.
Mục tiêu: Mô hình hóa định dạng lỗi JSON thống nhất toàn hệ thống:
    {"error": {"code": "VALIDATION_ERROR", "message": "Thông tin lỗi dễ hiểu."}}
"""

from typing import Any, Dict, Tuple

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Chi tiết mã lỗi và thông điệp lỗi gửi tới frontend."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Mã lỗi dạng SNAKE_CASE (VD: VALIDATION_ERROR, SERVICE_UNAVAILABLE)",
        examples=["VALIDATION_ERROR"],
    )
    message: str = Field(
        ...,
        description="Thông báo lỗi thân thiện, dễ hiểu cho người dùng cuối",
        examples=["Câu hỏi không được để trống hoặc vượt quá 2000 ký tự."],
    )


class ErrorResponse(BaseModel):
    """Bao bọc phản hồi lỗi chuẩn của toàn bộ API."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail = Field(..., description="Đối tượng chứa thông tin lỗi")


# Các mã lỗi hợp lệ theo từng HTTP status (dùng để sinh tài liệu OpenAPI).
ERROR_CODES_BY_STATUS: Dict[int, Tuple[str, ...]] = {
    404: ("NOT_FOUND",),
    405: ("METHOD_NOT_ALLOWED",),
    413: ("IMAGE_TOO_LARGE",),
    415: ("INVALID_IMAGE",),
    422: ("VALIDATION_ERROR",),
    500: ("INFERENCE_FAILED", "INTERNAL_SERVER_ERROR"),
    503: ("MODEL_UNAVAILABLE", "SERVICE_UNAVAILABLE"),
    504: ("INFERENCE_TIMEOUT",),
}


def error_responses(*statuses: int) -> Dict[int, Dict[str, Any]]:
    """Dict cho `responses=` của route, ví dụ error_responses(413, 415, 422, 503, 504).

    Raises:
        ValueError: nếu status chưa được khai báo trong ERROR_CODES_BY_STATUS.
    """
    unknown = [s for s in statuses if s not in ERROR_CODES_BY_STATUS]
    if unknown:
        raise ValueError(f"HTTP status chưa khai báo mã lỗi: {unknown}")

    return {
        status_code: {
            "model": ErrorResponse,
            "description": "Mã lỗi: " + " hoặc ".join(ERROR_CODES_BY_STATUS[status_code]),
        }
        for status_code in statuses
    }
