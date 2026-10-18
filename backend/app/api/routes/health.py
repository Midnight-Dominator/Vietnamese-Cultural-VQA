"""Endpoint kiểm tra sức khỏe hệ thống (Health Check).

Người phụ trách: Người 1 (Lead).
Mục tiêu: Cung cấp endpoint xác nhận backend đang chạy và sẵn sàng nhận request.
Lưu ý: Không kiểm tra database, không gọi model AI, không trả model ready khi chưa có model.
"""

from typing import Dict
from fastapi import APIRouter, status

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Kiểm tra trạng thái hoạt động của backend",
    response_description="Trạng thái của tiến trình backend",
)
async def check_health() -> Dict[str, str]:
    """Kiểm tra backend có đang chạy hay không.

    Trả về:
        {"status": "ok"} khi tiến trình hoạt động bình thường.
    """
    return {"status": "ok"}
