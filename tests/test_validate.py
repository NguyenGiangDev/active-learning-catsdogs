"""
tests/test_validate.py
Unit tests cho app/validate.py — kiểm tra validate_image() với các trường hợp hợp lệ và không hợp lệ.
"""

import io

import pytest
from PIL import Image

from app.validate import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_BYTES,
    MAX_IMAGE_DIMENSION,
    ImageValidationError,
    validate_image,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Tạo ảnh JPEG hợp lệ trong bộ nhớ."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Tạo ảnh PNG hợp lệ trong bộ nhớ."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(32, 128, 64))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Trường hợp hợp lệ ────────────────────────────────────────────────────────

class TestValidateImageHappyPath:
    def test_valid_jpeg(self) -> None:
        result = validate_image(_make_jpeg_bytes(), filename="cat.jpg", content_type="image/jpeg")
        assert result.mode == "RGB"

    def test_valid_png(self) -> None:
        result = validate_image(_make_png_bytes(), filename="dog.png", content_type="image/png")
        assert result.mode == "RGB"

    def test_no_filename_no_content_type(self) -> None:
        """Không có metadata vẫn phải decode được."""
        result = validate_image(_make_jpeg_bytes())
        assert result.mode == "RGB"

    def test_jpeg_extension_uppercase(self) -> None:
        result = validate_image(_make_jpeg_bytes(), filename="CAT.JPG")
        assert result.mode == "RGB"


# ── File rỗng / quá lớn ──────────────────────────────────────────────────────

class TestFileSizeValidation:
    def test_empty_file_raises(self) -> None:
        with pytest.raises(ImageValidationError, match="rỗng"):
            validate_image(b"")

    def test_oversized_file_raises(self) -> None:
        oversized = b"x" * (MAX_FILE_BYTES + 1)
        with pytest.raises(ImageValidationError, match="quá lớn"):
            validate_image(oversized, filename="big.jpg")


# ── Content-Type không hợp lệ ────────────────────────────────────────────────

class TestContentTypeValidation:
    def test_invalid_content_type_raises(self) -> None:
        with pytest.raises(ImageValidationError, match="Content-Type"):
            validate_image(_make_jpeg_bytes(), content_type="image/gif")

    def test_text_content_type_raises(self) -> None:
        with pytest.raises(ImageValidationError, match="Content-Type"):
            validate_image(_make_jpeg_bytes(), content_type="text/plain")


# ── Extension không hợp lệ ───────────────────────────────────────────────────

class TestExtensionValidation:
    def test_gif_extension_raises(self) -> None:
        with pytest.raises(ImageValidationError, match="mở rộng"):
            validate_image(_make_jpeg_bytes(), filename="cat.gif")

    def test_bmp_extension_raises(self) -> None:
        with pytest.raises(ImageValidationError, match="mở rộng"):
            validate_image(_make_jpeg_bytes(), filename="dog.bmp")


# ── Không decode được ────────────────────────────────────────────────────────

class TestDecodeValidation:
    def test_random_bytes_raises(self) -> None:
        with pytest.raises(ImageValidationError, match="đọc ảnh"):
            validate_image(b"\x00\x01\x02\x03" * 100, filename="fake.jpg")

    def test_truncated_jpeg_raises(self) -> None:
        data = _make_jpeg_bytes()[:50]  # cắt cụt
        with pytest.raises(ImageValidationError):
            validate_image(data, filename="truncated.jpg")


# ── Ảnh quá lớn (decompression bomb guard) ───────────────────────────────────

class TestDimensionValidation:
    def test_oversized_dimensions_raises(self) -> None:
        """Ảnh vượt MAX_IMAGE_DIMENSION phải bị từ chối."""
        big_size = MAX_IMAGE_DIMENSION + 1
        buf = io.BytesIO()
        # Dùng mode "L" (grayscale) để tạo ảnh lớn nhanh hơn trong test
        img = Image.new("L", (big_size, big_size), color=128)
        img.save(buf, format="PNG")
        with pytest.raises(ImageValidationError, match="quá lớn"):
            validate_image(buf.getvalue(), filename="huge.png")
