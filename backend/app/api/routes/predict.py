"""API hỏi đáp hình ảnh.

Route sở hữu ảnh từ read_image() và đóng ảnh trong finally.
Manager không giữ ảnh gốc sau khi predict() thoát: worker AI chạy tiếp sau timeout/hủy
dùng bản sao riêng do manager tạo và tự đóng.
"""

import logging

from anyio import CancelScope
from fastapi import APIRouter, File, Form, Request, UploadFile

from app.errors import AppError
from app.schemas.error import ErrorResponse
from app.schemas.prediction import PredictionResponse
from app.services import image_service, validation_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predict"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Gửi ảnh và câu hỏi để suy luận VQA",
    description=(
        "Nhận file ảnh, câu hỏi và mã model. "
        "Kiểm tra đầu vào, xử lý ảnh và gọi bộ quản lý model "
        "để trả về câu trả lời theo cấu trúc chung."
    ),
    responses={
        code: {"model": ErrorResponse, "description": description}
        for code, description in {
            413: "Ảnh vượt giới hạn dung lượng",
            415: "Ảnh không hợp lệ",
            422: "Dữ liệu đầu vào thiếu hoặc không hợp lệ",
            500: "Lỗi suy luận hoặc lỗi nội bộ",
            503: "Dịch vụ hoặc model chưa sẵn sàng",
            504: "Quá thời gian suy luận",
        }.items()
    },
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="File ảnh JPEG, PNG hoặc WEBP tĩnh"),
    question: str = Form(..., description="Nội dung câu hỏi"),
    model_id: str = Form(
        ...,
        description="Mã model được chọn: 'scratch' hoặc 'finetuned'",
    ),
) -> PredictionResponse:
    """Nhận yêu cầu hỏi đáp, gọi các dịch vụ và trả kết quả theo schema chung."""
    image = None

    try:
        # Gọi dịch vụ người 2 để kiểm tra và chuẩn hóa đầu vào.
        try:
            question = validation_service.validate_question(question)
            model_id = validation_service.validate_model_id(model_id)
        except ValueError as exc:
            # Chỉ chuyển ValueError ở bước kiểm tra đầu vào thành lỗi 422.
            # Không đưa nội dung ngoại lệ nội bộ ra phía người dùng.
            raise AppError(
                code="VALIDATION_ERROR",
                message="Câu hỏi hoặc mã model không hợp lệ.",
                status_code=422,
            ) from exc

        # Dùng cùng instance với /models và lifespan của ứng dụng.
        manager = getattr(request.app.state, "model_manager", None)
        if manager is None or not manager.initialized:
            raise AppError(
                code="SERVICE_UNAVAILABLE",
                message="Bộ quản lý model chưa sẵn sàng.",
                status_code=503,
            )

        # Gọi dịch vụ người 3 để đọc, kiểm tra và chuẩn hóa ảnh.
        image = await image_service.read_image(file)

        # Gọi bộ quản lý model của người 7.
        # Manager chịu trách nhiệm hàng đợi, timeout và điều phối worker AI.
        try:
            answer = await manager.predict(
                model_id=model_id,
                image=image,
                question=question,
            )
        except TimeoutError as exc:
            # Manager hiện ném TimeoutError khi yêu cầu hết hạn trong hàng đợi.
            # Chỉ đổi lỗi từ manager; không tạo timeout/hủy worker ở route.
            raise AppError(
                code="INFERENCE_TIMEOUT",
                message="Yêu cầu đã vượt quá thời gian suy luận cho phép.",
                status_code=504,
            ) from exc

        # Không trả thành công nếu đáp án rỗng hoặc sai kiểu dữ liệu.
        if not isinstance(answer, str) or not answer.strip():
            raise AppError(
                code="INFERENCE_FAILED",
                message="Model không trả về câu trả lời hợp lệ.",
                status_code=500,
            )

        # Trả kết quả theo schema chung của nhóm.
        return PredictionResponse(
            model_id=model_id,
            question=question,
            answer=answer,
        )

    except NotImplementedError as exc:
        # Báo chưa sẵn sàng nếu dịch vụ được gọi vẫn chưa triển khai.
        logger.warning(
            "Một dịch vụ phục vụ hỏi đáp chưa được triển khai.",
            exc_info=True,
        )
        raise AppError(
            code="SERVICE_UNAVAILABLE",
            message="Chức năng đang chờ tích hợp.",
            status_code=503,
        ) from exc

    finally:
        # Route luôn giải phóng ảnh mình sở hữu, kể cả lỗi/cancellation.
        # Manager không được giữ ảnh gốc sau khi predict() thoát; worker còn
        # chạy phải dùng bản sao riêng. Không bỏ cleanup để né lỗi manager.
        if image is not None:
            try:
                image.close()
            except Exception:
                # Lỗi dọn dẹp không được che mất lỗi xử lý ban đầu.
                logger.exception("Không thể đóng ảnh đã giải mã.")

        # Bảo vệ thao tác đóng file trước việc hủy yêu cầu qua AnyIO.
        # Không chuyển lỗi hủy yêu cầu thành phản hồi thành công.
        with CancelScope(shield=True):
            try:
                await file.close()
            except Exception:
                logger.exception("Không thể đóng file ảnh tải lên.")
