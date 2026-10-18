"""Xử lý ngoại lệ và chuẩn hóa cấu trúc phản hồi lỗi toàn hệ thống.

Người phụ trách: Người 1 (Lead).
Mục tiêu: Đảm bảo mọi lỗi (AppError, ValidationError, HTTPException, Exception không dự kiến)
đều trả về cấu trúc thống nhất:
    {"error": {"code": "...", "message": "..."}}
Không lộ stack trace hoặc đường dẫn nội bộ ra ngoài client.
Ghi log chi tiết tại server để debug mà không ghi nội dung file ảnh.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")


class AppError(Exception):
    """Lớp ngoại lệ cơ sở cho các lỗi nghiệp vụ trong hệ thống."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def format_error_payload(code: str, message: str) -> Dict[str, Any]:
    """Tạo payload phản hồi lỗi theo hợp đồng JSON đã thống nhất."""
    return {
        "error": {
            "code": code,
            "message": message,
        }
    }


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Xử lý ngoại lệ AppError."""
    logger.warning("AppError [%s]: %s (Path: %s)", exc.code, exc.message, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_payload(code=exc.code, message=exc.message),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Xử lý ngoại lệ kiểm tra dữ liệu đầu vào (422)."""
    # Rút gọn chi tiết lỗi thành thông báo dễ hiểu, không để lộ cấu trúc nội bộ nhạy cảm
    error_messages = []
    for err in exc.errors():
        loc = [str(item) for item in err.get("loc", []) if item not in ("body",)]
        loc_str = " -> ".join(loc)
        msg = err.get("msg", "Giá trị không hợp lệ")
        if loc_str:
            error_messages.append(f"{loc_str}: {msg}")
        else:
            error_messages.append(msg)

    message = "; ".join(error_messages) if error_messages else "Dữ liệu yêu cầu không hợp lệ."
    logger.info("Validation error trên %s: %s", request.url.path, message)

    return JSONResponse(
        status_code=422,
        content=format_error_payload(code="VALIDATION_ERROR", message=message),
    )



async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Xử lý HTTPException chuẩn của Starlette/FastAPI (bao gồm 404, 405, 503, ...)."""
    # Nếu chi tiết truyền vào đã là dict có code và message
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        code = str(exc.detail["code"])
        message = str(exc.detail["message"])
    elif isinstance(exc.detail, dict) and "error" in exc.detail:
        err_obj = exc.detail["error"]
        code = str(err_obj.get("code", "HTTP_ERROR"))
        message = str(err_obj.get("message", "Đã xảy ra lỗi HTTP."))
    else:
        # Ánh xạ status code sang mã lỗi chuẩn theo hợp đồng API
        http_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)
        http_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", None) or getattr(status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413)
        default_codes = {
            status.HTTP_404_NOT_FOUND: "NOT_FOUND",
            status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
            http_413: "IMAGE_TOO_LARGE",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: "INVALID_IMAGE",
            http_422: "VALIDATION_ERROR",
            status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
            status.HTTP_504_GATEWAY_TIMEOUT: "INFERENCE_TIMEOUT",
            status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL_SERVER_ERROR",
        }
        code = default_codes.get(exc.status_code, f"HTTP_{exc.status_code}")
        message = str(exc.detail) if exc.detail else "Yêu cầu không thể thực hiện."


    logger.warning("HTTP %d [%s] trên %s: %s", exc.status_code, code, request.url.path, message)

    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_payload(code=code, message=message),
        headers=exc.headers,
    )


from app.config import get_settings

settings = get_settings()


def get_cors_headers_for_request(request: Request) -> Dict[str, str]:
    """Tự động bổ sung CORS headers cho phản hồi lỗi khi origin hợp lệ.

    Đặc biệt quan trọng với lỗi 500 do ServerErrorMiddleware nằm ngoài CORSMiddleware
    trong pipeline ASGI của Starlette.
    """
    origin = request.headers.get("origin")
    headers: Dict[str, str] = {}
    if origin:
        normalized_origin = origin.strip().rstrip("/")
        if normalized_origin in settings.CORS_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
    return headers


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Xử lý ngoại lệ không dự kiến (500). Ghi log stack trace tại server, không trả ra FE."""
    # Log toàn bộ traceback trên server để điều tra lỗi
    logger.exception(
        "Ngoại lệ không dự kiến khi xử lý request %s %s: %s",
        request.method,
        request.url.path,
        str(exc),
    )
    headers = get_cors_headers_for_request(request)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=format_error_payload(
            code="INTERNAL_SERVER_ERROR",
            message="Đã xảy ra lỗi hệ thống không dự kiến. Vui lòng liên hệ quản trị viên.",
        ),
        headers=headers,
    )



def register_error_handlers(app: FastAPI) -> None:
    """Đăng ký toàn bộ exception handlers vào ứng dụng FastAPI."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
