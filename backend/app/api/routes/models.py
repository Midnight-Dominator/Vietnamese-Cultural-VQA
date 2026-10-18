"""Endpoint danh sách model AI.

GET /api/v1/models trả về `{"models": [{id, name, available, load_state}, ...]}` theo
docs/api_contract.md. Dữ liệu lấy trực tiếp từ ModelManager:
- available: model có thể phục vụ (đã nạp xong hoặc đủ file trọng số trên đĩa).
- load_state: unloaded | loading | ready | error.
"""

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.model import ModelInfo, ModelsResponse

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    status_code=status.HTTP_200_OK,
    summary="Lấy danh sách model AI được hỗ trợ",
    description="Trả về metadata và trạng thái nạp (load_state) thực tế của từng model.",
)
async def list_models(request: Request) -> ModelsResponse:
    manager = getattr(request.app.state, "model_manager", None)

    if manager is None or not manager.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Bộ quản lý model chưa sẵn sàng.",
            },
        )

    return ModelsResponse(models=[ModelInfo(**item) for item in manager.list_models()])
