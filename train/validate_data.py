"""
train/validate_data.py
Kiểm tra chất lượng dataset trước khi retrain.

Cách dùng:
    python -m train.validate_data --data-dir data/raw
    python -m train.validate_data --data-dir data/raw --min-samples 10 --max-ratio 5

Exit code:
    0 — dataset hợp lệ, có thể retrain
    1 — dataset KHÔNG hợp lệ (retrain bị dừng lại)
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# ── Ngưỡng mặc định ───────────────────────────────────────────────────────────
DEFAULT_MIN_SAMPLES_PER_CLASS: int = 5   # mỗi class cần ít nhất N ảnh
DEFAULT_MIN_TOTAL_SAMPLES: int = 10      # tổng dataset cần ít nhất N ảnh
DEFAULT_MAX_CLASS_IMBALANCE_RATIO: float = 10.0  # class lớn nhất / nhỏ nhất <= ratio
VALID_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGE_DIMENSION: int = 10  # ảnh phải rộng/cao ít nhất N pixel


# ── Kiểm tra từng ảnh ─────────────────────────────────────────────────────────

def validate_image_file(path: Path) -> tuple[bool, str]:
    """
    Mở file ảnh và kiểm tra tính hợp lệ.
    Trả về (is_valid, error_message).
    """
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return False, f"Phần mở rộng không hợp lệ: {path.suffix}"
    try:
        with Image.open(path) as img:
            img.verify()  # phát hiện file corrupt
        # Mở lại vì verify() đã đóng file
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                return False, f"Ảnh quá nhỏ: {w}x{h} px (tối thiểu {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})"
        return True, ""
    except UnidentifiedImageError:
        return False, "File không phải ảnh hợp lệ (UnidentifiedImageError)"
    except Exception as e:
        return False, f"Lỗi khi mở ảnh: {e}"


# ── Validate toàn bộ dataset ──────────────────────────────────────────────────

def validate_dataset(
    data_dir: str,
    min_samples_per_class: int = DEFAULT_MIN_SAMPLES_PER_CLASS,
    min_total_samples: int = DEFAULT_MIN_TOTAL_SAMPLES,
    max_class_imbalance_ratio: float = DEFAULT_MAX_CLASS_IMBALANCE_RATIO,
) -> bool:
    """
    Kiểm tra dataset theo cấu trúc ImageFolder.
    Trả về True nếu hợp lệ, False nếu có lỗi.
    """
    data_path = Path(data_dir)
    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 60)
    print("🔍 DATA VALIDATION")
    print(f"   data_dir : {data_path.resolve()}")
    print("=" * 60)

    # ── Kiểm tra thư mục tồn tại ─────────────────────────────────────────────
    if not data_path.exists():
        print(f"\n❌ FATAL: Thư mục không tồn tại: {data_path}")
        return False
    if not data_path.is_dir():
        print(f"\n❌ FATAL: Đường dẫn không phải thư mục: {data_path}")
        return False

    # ── Quét các class (subdirectory) ────────────────────────────────────────
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        print("\n❌ FATAL: Không tìm thấy class nào (thư mục con) trong data_dir")
        return False

    print(f"\n📁 Tìm thấy {len(class_dirs)} class(es): {[d.name for d in class_dirs]}\n")

    # ── Kiểm tra từng class ───────────────────────────────────────────────────
    counts_per_class: dict[str, int] = {}
    corrupt_files: list[str] = []

    for class_dir in class_dirs:
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        valid_count = 0
        for img_path in image_files:
            is_valid, err_msg = validate_image_file(img_path)
            if is_valid:
                valid_count += 1
            else:
                corrupt_files.append(f"  {img_path.relative_to(data_path)}: {err_msg}")

        counts_per_class[class_dir.name] = valid_count
        status = "✅" if valid_count >= min_samples_per_class else "❌"
        print(f"  {status} {class_dir.name:15s}: {valid_count:4d} ảnh hợp lệ", end="")
        if valid_count < min_samples_per_class:
            print(f"  ← cần ít nhất {min_samples_per_class}")
            errors.append(
                f"Class '{class_dir.name}' chỉ có {valid_count} ảnh "
                f"(tối thiểu {min_samples_per_class})"
            )
        else:
            print()

    # ── Kiểm tra file corrupt ─────────────────────────────────────────────────
    if corrupt_files:
        errors.append(f"{len(corrupt_files)} file ảnh bị lỗi/corrupt:")
        errors.extend(corrupt_files)

    # ── Kiểm tra tổng số ảnh ──────────────────────────────────────────────────
    total = sum(counts_per_class.values())
    print(f"\n  {'✅' if total >= min_total_samples else '❌'} Tổng: {total} ảnh hợp lệ", end="")
    if total < min_total_samples:
        print(f"  ← cần ít nhất {min_total_samples}")
        errors.append(f"Tổng dataset chỉ có {total} ảnh (tối thiểu {min_total_samples})")
    else:
        print()

    # ── Kiểm tra class imbalance ──────────────────────────────────────────────
    if len(counts_per_class) >= 2:
        max_count = max(counts_per_class.values())
        min_count = min(counts_per_class.values())
        ratio = max_count / max(min_count, 1)
        imbalance_ok = ratio <= max_class_imbalance_ratio
        print(f"  {'✅' if imbalance_ok else '⚠️ '} Class imbalance ratio: {ratio:.1f}x", end="")
        if not imbalance_ok:
            print(f"  ← vượt ngưỡng {max_class_imbalance_ratio}x")
            # Warning thay vì error — imbalance nặng mới fail
            if ratio > max_class_imbalance_ratio * 2:
                errors.append(
                    f"Class imbalance quá nặng: {ratio:.1f}x "
                    f"(ngưỡng warn {max_class_imbalance_ratio}x, ngưỡng fail {max_class_imbalance_ratio * 2}x)"
                )
            else:
                warnings.append(f"Class imbalance: {ratio:.1f}x (ngưỡng {max_class_imbalance_ratio}x)")
        else:
            print()

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if warnings:
        print(f"⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"   - {w}")

    if errors:
        print(f"❌ VALIDATION FAILED — {len(errors)} lỗi:")
        for e in errors:
            print(f"   - {e}")
        print("=" * 60)
        print("🚫 Retrain bị dừng. Hãy sửa data trước khi retrain.\n")
        return False
    else:
        print(f"✅ VALIDATION PASSED — Dataset hợp lệ!")
        print(f"   Tổng: {total} ảnh | Classes: {list(counts_per_class.keys())}")
        print("=" * 60 + "\n")
        return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate dataset trước khi retrain"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Thư mục dataset (ImageFolder format)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES_PER_CLASS,
        help=f"Số ảnh tối thiểu mỗi class (mặc định: {DEFAULT_MIN_SAMPLES_PER_CLASS})"
    )
    parser.add_argument(
        "--min-total", type=int, default=DEFAULT_MIN_TOTAL_SAMPLES,
        help=f"Tổng số ảnh tối thiểu (mặc định: {DEFAULT_MIN_TOTAL_SAMPLES})"
    )
    parser.add_argument(
        "--max-ratio", type=float, default=DEFAULT_MAX_CLASS_IMBALANCE_RATIO,
        help=f"Tỷ lệ imbalance tối đa (mặc định: {DEFAULT_MAX_CLASS_IMBALANCE_RATIO})"
    )
    args = parser.parse_args()

    ok = validate_dataset(
        data_dir=args.data_dir,
        min_samples_per_class=args.min_samples,
        min_total_samples=args.min_total,
        max_class_imbalance_ratio=args.max_ratio,
    )
    sys.exit(0 if ok else 1)
