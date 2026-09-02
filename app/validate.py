"""
app/validate.py
Validate file ảnh upload: định dạng, kích thước, decode an toàn.
Hàm này không có I/O phụ (stateless, dễ test).
"""

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# ── Hằng số cấu hình ─────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png"})
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
MAX_FILE_BYTES: int = 10 * 1024 * 1024   # 10 MB
MAX_IMAGE_DIMENSION: int = 4096          # pixel — tránh decompression bomb


class ImageValidationError(ValueError):
    """Lỗi validation ảnh, dùng để trả 400 Bad Request."""


def validate_image(data: bytes, filename: str = "", content_type: str = "") -> Image.Image:
    """
    Kiểm tra và decode bytes thành PIL.Image.

    Args:
        data: Raw bytes của file upload.
        filename: Tên file gốc (dùng để kiểm tra extension).
        content_type: MIME type từ Content-Type header.

    Returns:
        PIL.Image ở chế độ RGB, sẵn sàng đưa vào model.

    Raises:
        ImageValidationError: Khi file không hợp lệ.
    """
    # 1. Kiểm tra kích thước file
    if len(data) == 0:
        raise ImageValidationError("File rỗng.")
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError(
            f"File quá lớn ({len(data) / 1024 / 1024:.1f} MB). Giới hạn: {MAX_FILE_BYTES // 1024 // 1024} MB."
        )

    # 2. Kiểm tra content type (nếu có)
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"Content-Type không hợp lệ: {content_type!r}. Chấp nhận: {sorted(ALLOWED_CONTENT_TYPES)}."
        )

    # 3. Kiểm tra extension (nếu có tên file)
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Phần mở rộng không hợp lệ: {ext!r}. Chấp nhận: {sorted(ALLOWED_EXTENSIONS)}."
            )

    # 4. Decode an toàn (bắt mọi lỗi Pillow)
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()            # kiểm tra tính toàn vẹn mà không decode đầy đủ
    except (UnidentifiedImageError, Exception) as exc:
        raise ImageValidationError(f"Không thể đọc ảnh: {exc}") from exc

    # Mở lại sau verify() — Pillow yêu cầu mở lại sau khi verify
    img = Image.open(io.BytesIO(data))

    # 5. Kiểm tra kích thước pixel (decompression bomb guard)
    w, h = img.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Ảnh quá lớn ({w}x{h}). Giới hạn: {MAX_IMAGE_DIMENSION}px mỗi chiều."
        )

    # Chuẩn hóa về RGB (bỏ alpha, palette, v.v.)
    return img.convert("RGB")
