"""Điểm khởi chạy ứng dụng FastAPI chính (Application Entry Point).

Mục tiêu:
- Khởi tạo FastAPI app thông qua hàm factory create_app().
- Thiết lập middleware CORS (allow_credentials=False, không wildcard).
- Đăng ký hệ thống xử lý lỗi tập trung.
- Gắn router tổng /api/v1.
- Quản lý vòng đời ứng dụng (ModelManager startup/shutdown) bằng lifespan.
- Phục vụ giao diện web (thư mục frontend/) tại /ui/; "/" chuyển hướng sang đó.
- Khởi động trơn tru khi chưa có model weights: model chỉ được nạp khi có request đầu tiên.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.ai.model_manager import ModelManager, build_default_manager
from app.api.router import api_router
from app.config import get_settings
from app.errors import register_error_handlers

settings = get_settings()

# Thiết lập logging cơ bản
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Vòng đời ứng dụng: khởi động và dọn dẹp ModelManager.

    Không nạp trọng số lúc khởi động (trừ khi đặt PRELOAD_MODEL), nên server luôn lên được
    kể cả khi chưa có checkpoint; GET /models báo available=false cho model còn thiếu file.
    """
    logger.info("=== BẮT ĐẦU KHỞI ĐỘNG HỆ THỐNG: %s ===", settings.APP_NAME)
    logger.info("API Prefix: %s | Thiết bị AI: %s", settings.API_PREFIX, settings.AI_DEVICE)
    logger.info("Thư mục Scratch Model: %s", settings.resolved_scratch_model_path)
    logger.info("Thư mục Finetuned Model: %s", settings.resolved_finetuned_model_path)

    manager: ModelManager = app.state.model_manager
    await manager.startup()

    yield

    logger.info("=== ĐANG TẮT HỆ THỐNG: %s ===", settings.APP_NAME)
    await manager.shutdown()
    logger.info("Hoàn tất dọn dẹp tiến trình backend.")


def create_app(model_manager: ModelManager | None = None) -> FastAPI:
    """Tạo và cấu hình ứng dụng FastAPI hoàn chỉnh.

    model_manager: truyền manager tùy biến (dùng trong kiểm thử); mặc định là manager thật
    với hai model scratch và finetuned.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Backend dịch vụ Hỏi Đáp Bằng Hình Ảnh (Image VQA Demo) "
            "hỗ trợ 2 model: scratch và finetuned."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.model_manager = model_manager or build_default_manager(settings)

    # 1. Đăng ký Middleware CORS
    # Cho phép các origin đã cấu hình, cấm wildcard '*', allow_credentials=False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # 2. Đăng ký xử lý lỗi chung toàn hệ thống
    register_error_handlers(app)

    # 3. Đăng ký Router tổng hợp (chứa prefix /api/v1)
    app.include_router(api_router)

    # 4. Giao diện web tại /ui (không mount ở "/" để khỏi che các route khác thêm sau này).
    frontend_dir = settings.resolved_frontend_dir
    if settings.SERVE_FRONTEND:
        if frontend_dir.is_dir():
            app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")

            @app.get("/", include_in_schema=False)
            async def open_frontend() -> RedirectResponse:
                return RedirectResponse(url="/ui/")
        else:
            logger.warning("Không tìm thấy thư mục giao diện %s; chỉ phục vụ API.", frontend_dir)

    return app


# Biến app instance cho ASGI servers (Uvicorn)
app = create_app()
