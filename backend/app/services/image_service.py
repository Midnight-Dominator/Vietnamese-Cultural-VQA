"""Đọc UploadFile thành ảnh Pillow RGB theo hợp đồng của nhóm.

Người 3: kiểm tra dung lượng, magic bytes, ảnh tĩnh, dữ liệu hỏng và EXIF.
Module không ghi ảnh ra đĩa và không resize/normalize theo model.
Upload và ảnh trung gian được đóng ở đây; người gọi đóng ảnh trả về sau AI.
"""

from contextlib import closing
from io import BytesIO
import struct

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.errors import AppError

settings = get_settings()
_CHUNK_BYTES = 64 * 1024
_FORMATS = ("JPEG", "PNG", "WEBP")


def _invalid_image(message: str) -> AppError:
    return AppError(code="INVALID_IMAGE", message=message, status_code=415)


def _detect_format(data: bytes) -> str:
    """Kiểm tra chữ ký file, không tin tên file hoặc Content-Type."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    raise _invalid_image("Chỉ hỗ trợ dữ liệu ảnh JPEG, PNG hoặc WEBP hợp lệ.")


def _check_header(image: Image.Image, expected_format: str) -> None:
    if image.format != expected_format:
        raise _invalid_image("Định dạng ảnh không khớp với dữ liệu file.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise _invalid_image("Kích thước ảnh không hợp lệ.")
    # Giữ cơ chế bảo vệ decompression bomb mặc định của Pillow.
    pixel_limit = Image.MAX_IMAGE_PIXELS
    if pixel_limit is not None and width * height > pixel_limit:
        raise _invalid_image("Ảnh vượt giới hạn số pixel giải mã an toàn.")
    if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
        raise _invalid_image("Chỉ hỗ trợ ảnh tĩnh một khung hình.")


def _decode_image(data: bytes) -> Image.Image:
    """Giải mã đồng bộ trong RAM; trả ảnh độc lập với các buffer đã đóng."""
    expected_format = _detect_format(data)
    try:
        with BytesIO(data) as buffer:
            with closing(Image.open(buffer, formats=_FORMATS)) as source:
                _check_header(source, expected_format)
                source.verify()

        # Sau verify() phải mở lại để load toàn bộ pixel.
        with BytesIO(data) as buffer:
            with closing(Image.open(buffer, formats=_FORMATS)) as source:
                _check_header(source, expected_format)
                source.load()
                with closing(ImageOps.exif_transpose(source)) as oriented:
                    if "A" in oriented.getbands() or "transparency" in oriented.info:
                        with closing(oriented.convert("RGBA")) as rgba:
                            with closing(Image.new("RGBA", rgba.size, "white")) as background:
                                with closing(Image.alpha_composite(background, rgba)) as merged:
                                    result = merged.convert("RGB")
                    else:
                        result = oriented.convert("RGB")
                    try:
                        result.load()
                        result.info.clear()
                        return result
                    except BaseException:
                        result.close()
                        raise
    except AppError:
        raise
    except Image.DecompressionBombError as exc:
        raise _invalid_image("Ảnh vượt giới hạn giải mã an toàn.") from exc
    except UnidentifiedImageError as exc:
        raise _invalid_image("Dữ liệu không phải ảnh hợp lệ hoặc file bị hỏng.") from exc
    except (OSError, SyntaxError, ValueError, EOFError, struct.error) as exc:
        raise _invalid_image("File ảnh bị hỏng hoặc không thể giải mã.") from exc


async def read_image(file: UploadFile) -> Image.Image:
    """Đọc từ vị trí hiện tại, kiểm tra và trả ảnh RGB đã load hoàn toàn.

    Dung lượng lấy từ settings.MAX_UPLOAD_BYTES (mặc định 10 MiB).
    Đọc tối đa giới hạn + 1 byte để phát hiện vượt dung lượng; luôn đóng file.
    Gọi ngay với UploadFile mới; nếu đã đọc file, caller cần seek(0) trước.
    Người gọi sở hữu ảnh trả về và đóng ảnh sau khi suy luận thật sự kết thúc.
    """
    max_bytes = settings.MAX_UPLOAD_BYTES
    try:
        with BytesIO() as buffer:
            while True:
                remaining = max_bytes + 1 - buffer.tell()
                chunk = await file.read(min(_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                buffer.write(chunk)
                if buffer.tell() > max_bytes:
                    raise AppError(
                        code="IMAGE_TOO_LARGE",
                        message=f"Dung lượng ảnh vượt giới hạn {max_bytes} byte.",
                        status_code=413,
                    )
            data = buffer.getvalue()
    finally:
        await file.close()

    if not data:
        raise _invalid_image("File ảnh rỗng.")

    # Giải mã Pillow không chặn event loop của endpoint async.
    return await run_in_threadpool(_decode_image, data)
