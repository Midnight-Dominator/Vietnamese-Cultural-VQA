"""Dịch vụ kiểm tra và xác thực nghiệp vụ (Validation Service).

Người phụ trách triển khai chính: Người 2.
Mục tiêu: Cung cấp các hàm kiểm tra tính hợp lệ của câu hỏi và model_id theo ràng buộc dự án.

Hợp đồng giao diện:
- validate_question(question: str) -> str:
    + Chuẩn hóa Unicode về NFC và loại bỏ khoảng trắng thừa (strip).
    + Kiểm tra độ dài từ 1 đến 2000 ký tự (lấy cấu hình từ Settings.MAX_QUESTION_LENGTH).
    + Ném AppError (VALIDATION_ERROR, HTTP 422) nếu không hợp lệ.
    + Trả về chuỗi câu hỏi đã được làm sạch.
- validate_model_id(model_id: str) -> str:
    + Kiểm tra model_id phải thuộc tập hợp cho phép: {'scratch', 'finetuned'} (MODEL_IDS).
    + Ném AppError (VALIDATION_ERROR, HTTP 422) nếu model_id không hợp lệ.
    + Trả về model_id hợp lệ.

Trạng thái hiện tại:
- Đã triển khai đầy đủ cả hai hàm (không còn NotImplementedError).
"""

import unicodedata

from app.config import get_settings
from app.errors import AppError
from app.schemas.model import MODEL_IDS


def _validation_error(message: str) -> AppError:
    """Tạo AppError chuẩn cho lỗi xác thực đầu vào (VALIDATION_ERROR, HTTP 422)."""
    return AppError(code="VALIDATION_ERROR", message=message, status_code=422)


def validate_question(question: str) -> str:
    """Xác thực và chuẩn hóa câu hỏi người dùng nhập.

    Các bước: chuẩn hóa NFC, strip hai đầu, kiểm tra không rỗng và
    độ dài <= settings.MAX_QUESTION_LENGTH (tính sau chuẩn hóa).

    Chuẩn hóa trước khi đếm để tiếng Việt dạng tổ hợp (NFD) không bị tính dài oan.

    Returns:
        Câu hỏi sau khi chuẩn hóa.

    Raises:
        AppError: VALIDATION_ERROR (422) nếu không phải chuỗi, rỗng hoặc quá dài.
    """
    if not isinstance(question, str):
        raise _validation_error("Câu hỏi phải là chuỗi văn bản.")

    normalized = unicodedata.normalize("NFC", question).strip()
    if not normalized:
        raise _validation_error("Câu hỏi không được để trống.")

    # Đọc cấu hình tại thời điểm gọi để không bị khóa cứng giá trị lúc import module.
    max_length = get_settings().MAX_QUESTION_LENGTH
    if len(normalized) > max_length:
        raise _validation_error(f"Câu hỏi không được vượt quá {max_length} ký tự.")

    return normalized


def validate_model_id(model_id: str) -> str:
    """Xác thực mã model được chọn.

    Chỉ nhận đúng id trong MODEL_IDS ('scratch', 'finetuned'); không sửa id sai
    và không in lại giá trị người dùng gửi.

    Returns:
        model_id hợp lệ.

    Raises:
        AppError: VALIDATION_ERROR (422) nếu model_id không hợp lệ.
    """
    if not isinstance(model_id, str) or model_id not in MODEL_IDS:
        raise _validation_error("model_id không hợp lệ. Chỉ chấp nhận: " + ", ".join(MODEL_IDS) + ".")
    return model_id
