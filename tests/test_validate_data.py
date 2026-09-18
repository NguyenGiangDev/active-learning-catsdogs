"""
tests/test_validate_data.py
Unit tests cho train/validate_data.py
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from train.validate_data import validate_dataset, validate_image_file


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dummy_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Tạo ảnh RGB giả để dùng trong test."""
    img = Image.new("RGB", size, color=(128, 64, 32))
    img.save(path)


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """Dataset hợp lệ: 2 class, mỗi class 6 ảnh."""
    for cls in ("Cat", "Dog"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(6):
            make_dummy_image(cls_dir / f"img{i:03d}.jpg")
    return tmp_path


@pytest.fixture
def imbalanced_dataset(tmp_path: Path) -> Path:
    """Dataset imbalance nhẹ: Cat=6, Dog=60 (ratio=10x — ngay ngưỡng warn)."""
    (tmp_path / "Cat").mkdir()
    (tmp_path / "Dog").mkdir()
    for i in range(6):
        make_dummy_image(tmp_path / "Cat" / f"img{i:03d}.jpg")
    for i in range(60):
        make_dummy_image(tmp_path / "Dog" / f"img{i:03d}.jpg")
    return tmp_path


@pytest.fixture
def severely_imbalanced_dataset(tmp_path: Path) -> Path:
    """Dataset imbalance nặng: Cat=1, Dog=100 (ratio=100x — vượt fail threshold)."""
    (tmp_path / "Cat").mkdir()
    (tmp_path / "Dog").mkdir()
    make_dummy_image(tmp_path / "Cat" / "img000.jpg")
    for i in range(100):
        make_dummy_image(tmp_path / "Dog" / f"img{i:03d}.jpg")
    return tmp_path


# ── Tests: validate_image_file ─────────────────────────────────────────────────

class TestValidateImageFile:
    def test_valid_jpg(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.jpg"
        make_dummy_image(p)
        ok, msg = validate_image_file(p)
        assert ok is True
        assert msg == ""

    def test_valid_png(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.png"
        make_dummy_image(p)
        ok, msg = validate_image_file(p)
        assert ok is True

    def test_invalid_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("not an image")
        ok, msg = validate_image_file(p)
        assert ok is False
        assert "mở rộng" in msg

    def test_corrupt_file(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"this is not a valid image")
        ok, msg = validate_image_file(p)
        assert ok is False

    def test_too_small_image(self, tmp_path: Path) -> None:
        p = tmp_path / "tiny.jpg"
        make_dummy_image(p, size=(5, 5))
        ok, msg = validate_image_file(p)
        assert ok is False
        assert "nhỏ" in msg


# ── Tests: validate_dataset ────────────────────────────────────────────────────

class TestValidateDataset:
    def test_valid_dataset_passes(self, valid_dataset: Path) -> None:
        result = validate_dataset(str(valid_dataset), min_samples_per_class=5, min_total_samples=10)
        assert result is True

    def test_missing_directory_fails(self, tmp_path: Path) -> None:
        result = validate_dataset(str(tmp_path / "nonexistent"))
        assert result is False

    def test_empty_directory_fails(self, tmp_path: Path) -> None:
        result = validate_dataset(str(tmp_path))
        assert result is False

    def test_too_few_samples_per_class_fails(self, tmp_path: Path) -> None:
        """Class chỉ có 2 ảnh nhưng yêu cầu tối thiểu 5."""
        (tmp_path / "Cat").mkdir()
        (tmp_path / "Dog").mkdir()
        for i in range(2):
            make_dummy_image(tmp_path / "Cat" / f"img{i}.jpg")
        for i in range(6):
            make_dummy_image(tmp_path / "Dog" / f"img{i}.jpg")
        result = validate_dataset(str(tmp_path), min_samples_per_class=5, min_total_samples=5)
        assert result is False

    def test_too_few_total_fails(self, tmp_path: Path) -> None:
        """Tổng chỉ 4 ảnh nhưng yêu cầu 10."""
        (tmp_path / "Cat").mkdir()
        (tmp_path / "Dog").mkdir()
        for i in range(2):
            make_dummy_image(tmp_path / "Cat" / f"img{i}.jpg")
            make_dummy_image(tmp_path / "Dog" / f"img{i}.jpg")
        result = validate_dataset(
            str(tmp_path),
            min_samples_per_class=1,
            min_total_samples=10,
        )
        assert result is False

    def test_moderate_imbalance_passes(self, imbalanced_dataset: Path) -> None:
        """Imbalance 10x vẫn pass (dưới ngưỡng fail 20x)."""
        result = validate_dataset(
            str(imbalanced_dataset),
            min_samples_per_class=5,
            min_total_samples=10,
            max_class_imbalance_ratio=10.0,
        )
        assert result is True

    def test_severe_imbalance_fails(self, severely_imbalanced_dataset: Path) -> None:
        """Imbalance 100x vượt ngưỡng fail (2 × max_ratio = 20x)."""
        result = validate_dataset(
            str(severely_imbalanced_dataset),
            min_samples_per_class=1,
            min_total_samples=5,
            max_class_imbalance_ratio=10.0,
        )
        assert result is False

    def test_corrupt_images_counted_as_invalid(self, tmp_path: Path) -> None:
        """File corrupt không được tính vào valid_count → fail."""
        (tmp_path / "Cat").mkdir()
        (tmp_path / "Dog").mkdir()
        # Cat: 3 ảnh hợp lệ + 2 corrupt → valid_count=3 < min=5
        for i in range(3):
            make_dummy_image(tmp_path / "Cat" / f"ok{i}.jpg")
        for i in range(2):
            (tmp_path / "Cat" / f"bad{i}.jpg").write_bytes(b"not_an_image")
        for i in range(6):
            make_dummy_image(tmp_path / "Dog" / f"img{i}.jpg")

        result = validate_dataset(str(tmp_path), min_samples_per_class=5, min_total_samples=5)
        assert result is False

    def test_custom_min_samples(self, valid_dataset: Path) -> None:
        """Tăng min_samples_per_class lên 20 → fail vì chỉ có 6 ảnh/class."""
        result = validate_dataset(str(valid_dataset), min_samples_per_class=20)
        assert result is False
