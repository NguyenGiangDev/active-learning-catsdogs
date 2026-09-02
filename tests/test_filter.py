"""
tests/test_filter.py
Unit tests cho app/filter.py — bao phủ trường hợp biên và cả hai phía ngưỡng.
"""

import pytest

from app.filter import DEFAULT_THRESHOLD, is_uncertain


class TestIsUncertain:
    """Tests cho hàm is_uncertain()."""

    # ── Dưới ngưỡng → bất định ────────────────────────────────────────────────
    def test_well_below_threshold(self) -> None:
        assert is_uncertain(0.0) is True

    def test_below_threshold(self) -> None:
        assert is_uncertain(0.50) is True

    def test_just_below_threshold(self) -> None:
        assert is_uncertain(0.79) is True

    # ── Đúng bằng ngưỡng → vẫn bất định (≤) ─────────────────────────────────
    def test_exactly_at_threshold(self) -> None:
        """confidence == threshold vẫn là bất định theo quy ước ≤."""
        assert is_uncertain(DEFAULT_THRESHOLD) is True

    def test_exactly_at_custom_threshold(self) -> None:
        assert is_uncertain(0.90, threshold=0.90) is True

    # ── Trên ngưỡng → đủ tự tin ──────────────────────────────────────────────
    def test_just_above_threshold(self) -> None:
        assert is_uncertain(0.81) is False

    def test_well_above_threshold(self) -> None:
        assert is_uncertain(1.0) is False

    def test_high_confidence(self) -> None:
        assert is_uncertain(0.99) is False

    # ── Custom threshold ──────────────────────────────────────────────────────
    def test_custom_threshold_certain(self) -> None:
        assert is_uncertain(0.70, threshold=0.60) is False

    def test_custom_threshold_uncertain(self) -> None:
        assert is_uncertain(0.55, threshold=0.60) is True

    # ── Giá trị không hợp lệ → ValueError ────────────────────────────────────
    def test_negative_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            is_uncertain(-0.01)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            is_uncertain(1.001)
