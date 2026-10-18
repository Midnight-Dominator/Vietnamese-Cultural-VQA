"""Pydantic schemas cho API danh sách model AI.

Người phụ trách triển khai chính: Người 2.
Mục tiêu: Định nghĩa schema cấu trúc thông tin model trả về cho client.

Hợp đồng dữ liệu thống nhất:
- GET /api/v1/models
- Response HTTP 200 dự kiến:
    {
        "models": [
            {
                "id": "scratch",
                "name": "Model tự xây",
                "available": false,
                "load_state": "unloaded"
            },
            ...
        ]
    }
- load_state: unloaded | loading | ready | error
"""

from typing import List, Literal, Tuple, get_args

from pydantic import BaseModel, ConfigDict, Field

# Nguồn sự thật duy nhất cho id model và trạng thái nạp; các schema khác import lại từ đây.
ModelId = Literal["scratch", "finetuned"]
LoadState = Literal["unloaded", "loading", "ready", "error"]

MODEL_IDS: Tuple[str, ...] = get_args(ModelId)


# TODO (Người 2): Hoàn thiện và cập nhật schema theo thực tế khi Người 7 bàn giao ModelManager
class ModelInfo(BaseModel):
    """Thông tin chi tiết của từng model AI."""

    model_config = ConfigDict(extra="forbid")

    id: ModelId = Field(
        ...,
        description="Mã định danh duy nhất của model",
        examples=["scratch"],
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Tên hiển thị thân thiện với người dùng",
        examples=["Model tự xây"],
    )
    available: bool = Field(
        ...,
        description="Model có sẵn sàng phục vụ suy luận hay không",
    )
    load_state: LoadState = Field(
        ...,
        description="Trạng thái nạp trọng số hiện tại của model",
    )


class ModelsResponse(BaseModel):
    """Danh sách các model AI được backend hỗ trợ."""

    model_config = ConfigDict(extra="forbid")

    models: List[ModelInfo] = Field(..., description="Danh sách model hỗ trợ")


# Tên cũ trong khung nền, giữ để code đã import không bị vỡ.
ModelItem = ModelInfo
ModelListResponse = ModelsResponse
