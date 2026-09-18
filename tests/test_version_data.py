"""
tests/test_version_data.py
Unit tests cho train/version_data.py
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image

from train.version_data import (
    count_images_per_class,
    load_versions_registry,
    save_versions_registry,
    version_data,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dummy_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path)


def make_dataset(root: Path, counts: dict[str, int]) -> None:
    """Tạo dataset ImageFolder giả với số ảnh theo từng class."""
    for cls, n in counts.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            make_dummy_image(cls_dir / f"img{i:04d}.jpg")


# ── Tests: count_images_per_class ─────────────────────────────────────────────

class TestCountImagesPerClass:
    def test_counts_correctly(self, tmp_path: Path) -> None:
        make_dataset(tmp_path, {"Cat": 10, "Dog": 15})
        counts = count_images_per_class(tmp_path)
        assert counts == {"Cat": 10, "Dog": 15}

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        (tmp_path / "Cat").mkdir()
        make_dummy_image(tmp_path / "Cat" / "img.jpg")
        (tmp_path / "Cat" / "readme.txt").write_text("ignore me")
        counts = count_images_per_class(tmp_path)
        assert counts["Cat"] == 1

    def test_empty_class_returns_zero(self, tmp_path: Path) -> None:
        (tmp_path / "Cat").mkdir()
        counts = count_images_per_class(tmp_path)
        assert counts["Cat"] == 0

    def test_sorted_class_order(self, tmp_path: Path) -> None:
        make_dataset(tmp_path, {"Dog": 3, "Cat": 5})
        counts = count_images_per_class(tmp_path)
        assert list(counts.keys()) == ["Cat", "Dog"]  # sorted alphabetically


# ── Tests: registry load/save ─────────────────────────────────────────────────

class TestVersionsRegistry:
    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        registry = load_versions_registry(tmp_path / "nonexistent.json")
        assert registry == []

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "versions.json"
        data = [{"version": "dataset-v1", "total_samples": 100}]
        save_versions_registry(path, data)
        loaded = load_versions_registry(path)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "versions.json"
        save_versions_registry(path, [])
        assert path.exists()

    def test_registry_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "versions.json"
        data = [{"version": "dataset-v1"}, {"version": "dataset-v2"}]
        save_versions_registry(path, data)
        # Đọc raw và parse lại để đảm bảo JSON hợp lệ
        raw = path.read_text()
        parsed = json.loads(raw)
        assert len(parsed) == 2


# ── Tests: version_data (integration, mock DVC/git) ──────────────────────────

class TestVersionData:
    """
    Mock các subprocess call (dvc, git) để test logic Python thuần,
    không phụ thuộc vào DVC/git được cài.
    """

    @patch("train.version_data.dvc_push", return_value=False)
    @patch("train.version_data.git_commit_and_tag", return_value=False)
    @patch("train.version_data.dvc_add", return_value=False)
    def test_creates_registry_entry(
        self,
        mock_dvc_add: MagicMock,
        mock_git: MagicMock,
        mock_push: MagicMock,
        tmp_path: Path,
    ) -> None:
        make_dataset(tmp_path, {"Cat": 5, "Dog": 8})
        registry_path = tmp_path / "versions.json"

        with patch("train.version_data.VERSIONS_REGISTRY", str(registry_path)):
            tag = version_data(
                data_dir=str(tmp_path),
                run_id="abc123def456",
                dvc_remote="localremote",
            )

        assert tag == "dataset-v1"
        versions = load_versions_registry(registry_path)
        assert len(versions) == 1
        entry = versions[0]
        assert entry["version"] == "dataset-v1"
        assert entry["version_num"] == 1
        assert entry["total_samples"] == 13
        assert entry["counts_per_class"] == {"Cat": 5, "Dog": 8}
        assert entry["git_run_id"] == "abc123def456"

    @patch("train.version_data.dvc_push", return_value=False)
    @patch("train.version_data.git_commit_and_tag", return_value=False)
    @patch("train.version_data.dvc_add", return_value=False)
    def test_version_increments_with_each_call(
        self,
        mock_dvc_add: MagicMock,
        mock_git: MagicMock,
        mock_push: MagicMock,
        tmp_path: Path,
    ) -> None:
        make_dataset(tmp_path, {"Cat": 5, "Dog": 5})
        registry_path = tmp_path / "versions.json"

        with patch("train.version_data.VERSIONS_REGISTRY", str(registry_path)):
            tag1 = version_data(str(tmp_path), "run001", "localremote")
            tag2 = version_data(str(tmp_path), "run002", "localremote")
            tag3 = version_data(str(tmp_path), "run003", "localremote")

        assert tag1 == "dataset-v1"
        assert tag2 == "dataset-v2"
        assert tag3 == "dataset-v3"

        versions = load_versions_registry(registry_path)
        assert len(versions) == 3

    @patch("train.version_data.dvc_push", return_value=True)
    @patch("train.version_data.git_commit_and_tag", return_value=True)
    @patch("train.version_data.dvc_add", return_value=True)
    def test_records_dvc_remote_in_entry(
        self,
        mock_dvc_add: MagicMock,
        mock_git: MagicMock,
        mock_push: MagicMock,
        tmp_path: Path,
    ) -> None:
        make_dataset(tmp_path, {"Cat": 5, "Dog": 5})
        registry_path = tmp_path / "versions.json"

        with patch("train.version_data.VERSIONS_REGISTRY", str(registry_path)):
            version_data(str(tmp_path), "sha999", "s3remote")

        entry = load_versions_registry(registry_path)[0]
        assert entry["dvc_remote"] == "s3remote"
        assert entry["dvc_push_ok"] is True
        assert entry["git_tag_ok"] is True

    @patch("train.version_data.dvc_push", return_value=False)
    @patch("train.version_data.git_commit_and_tag", return_value=False)
    @patch("train.version_data.dvc_add", return_value=False)
    def test_dvc_add_called_with_correct_path(
        self,
        mock_dvc_add: MagicMock,
        mock_git: MagicMock,
        mock_push: MagicMock,
        tmp_path: Path,
    ) -> None:
        make_dataset(tmp_path, {"Cat": 5, "Dog": 5})
        registry_path = tmp_path / "versions.json"

        with patch("train.version_data.VERSIONS_REGISTRY", str(registry_path)):
            version_data(str(tmp_path), "run001", "localremote")

        mock_dvc_add.assert_called_once_with(tmp_path)

    @patch("train.version_data.dvc_push", return_value=False)
    @patch("train.version_data.git_commit_and_tag", return_value=False)
    @patch("train.version_data.dvc_add", return_value=False)
    def test_entry_has_timestamp(
        self,
        mock_dvc_add: MagicMock,
        mock_git: MagicMock,
        mock_push: MagicMock,
        tmp_path: Path,
    ) -> None:
        make_dataset(tmp_path, {"Cat": 5, "Dog": 5})
        registry_path = tmp_path / "versions.json"

        with patch("train.version_data.VERSIONS_REGISTRY", str(registry_path)):
            version_data(str(tmp_path), "run001", "localremote")

        entry = load_versions_registry(registry_path)[0]
        assert "timestamp" in entry
        # ISO 8601 format check
        assert "T" in entry["timestamp"]
        assert "+00:00" in entry["timestamp"] or "Z" in entry["timestamp"] or "UTC" in entry["timestamp"]
