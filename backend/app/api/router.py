"""Router tổng tập hợp các router con của ứng dụng.

Người phụ trách: Người 1 (Lead).
Mục tiêu: Đăng ký các router thành phần (health, models, predict) vào router chính.
Quy tắc:
- Chỉ đặt prefix /api/v1 một lần duy nhất tại router tổng này.
- Các route con giữ đường dẫn tương đối (/health, /models, /predict).
- Tránh tuyệt đối việc lặp URL như /api/v1/api/v1.
"""

from fastapi import APIRouter
from app.api.routes import health, models, predict
from app.config import get_settings

settings = get_settings()

api_router = APIRouter(prefix=settings.API_PREFIX)

# Đăng ký các router thành phần
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(predict.router)
