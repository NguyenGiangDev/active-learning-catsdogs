"""
app/filter.py
Bộ lọc độ bất định — lõi của vòng lặp Active Learning.

Hàm is_uncertain() là pure function (không có I/O, dễ unit test).
"""

# ── Hằng số cấu hình ─────────────────────────────────────────────────────────
DEFAULT_THRESHOLD: float = 0.80  # confidence ≤ ngưỡng này → coi là bất định


def is_uncertain(confidence: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """
    Xác định xem dự đoán có đủ tự tin hay không.

    Args:
        confidence: Điểm tự tin của model, trong khoảng [0.0, 1.0].
        threshold: Ngưỡng tự tin (mặc định 0.80).
                   Confidence **≤** threshold → bất định (cần gán nhãn lại).

    Returns:
        True  → model bất định, nên gửi ảnh cho con người xem xét.
        False → model đủ tự tin, trả kết quả trực tiếp.

    Examples:
        >>> is_uncertain(0.75)
        True
        >>> is_uncertain(0.80)   # đúng bằng ngưỡng → vẫn bất định
        True
        >>> is_uncertain(0.81)
        False
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence phải trong [0, 1], nhận được: {confidence}")
    return confidence <= threshold
