"""
train/version_data.py
Đánh version cho dataset bằng DVC trước mỗi lần retrain.

Cách dùng:
    python -m train.version_data --data-dir data/raw --run-id abc123
    python -m train.version_data --data-dir data/raw --run-id $GITHUB_SHA

Luồng thực hiện:
    1. Đọc registry data/dataset_versions.json → tính version tiếp theo
    2. Tạo version tag: dataset-v{N}
    3. dvc add data/raw          → cập nhật data/raw.dvc (content hash)
    4. git add + git commit       → đóng băng con trỏ vào git
    5. git tag dataset-v{N}       → đặt tên snapshot
    6. dvc push --remote {remote} → upload data lên S3 / local remote
    7. Cập nhật data/dataset_versions.json

Biến môi trường:
    DVC_REMOTE  — tên DVC remote cần push (mặc định: localremote)
                  Đặt DVC_REMOTE=s3remote khi deploy trên AWS
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VALID_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VERSIONS_REGISTRY: str = "data/dataset_versions.json"
DEFAULT_DVC_REMOTE: str = "localremote"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Chạy shell command, in output, raise nếu lỗi (khi check=True)."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=capture,
        text=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def count_images_per_class(data_dir: Path) -> dict[str, int]:
    """Đếm số ảnh hợp lệ theo từng class (subdirectory)."""
    counts: dict[str, int] = {}
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        counts[class_dir.name] = n
    return counts


def load_versions_registry(registry_path: Path) -> list[dict]:
    """Đọc file registry. Trả về list rỗng nếu chưa tồn tại."""
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return []


def save_versions_registry(registry_path: Path, versions: list[dict]) -> None:
    """Ghi registry ra file JSON."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(versions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ── Bước DVC / Git ────────────────────────────────────────────────────────────

def dvc_add(data_dir: Path) -> bool:
    """
    Chạy `dvc add <data_dir>`.
    Trả về True nếu thành công, False nếu DVC chưa được cài / lỗi (demo mode).
    """
    try:
        run_cmd(["dvc", "add", str(data_dir)])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ⚠️  dvc add bỏ qua (demo mode): {e}")
        return False


def git_commit_and_tag(data_dir: Path, tag: str, message: str) -> bool:
    """
    Commit data/raw.dvc + dataset_versions.json vào git rồi tạo tag.
    Trả về True nếu thành công.
    """
    dvc_file = Path(f"{data_dir}.dvc")
    try:
        # Cấu hình git identity tối thiểu cho CI (nếu chưa có)
        run_cmd(["git", "config", "user.email"], capture=True, check=False)
        result = run_cmd(["git", "config", "user.email"], capture=True, check=False)
        if not result.stdout.strip():
            run_cmd(["git", "config", "user.email", "ci-bot@github-actions"])
            run_cmd(["git", "config", "user.name", "GitHub Actions"])

        files_to_add = [VERSIONS_REGISTRY]
        if dvc_file.exists():
            files_to_add.append(str(dvc_file))

        run_cmd(["git", "add"] + files_to_add)

        # Kiểm tra có gì để commit không
        status = run_cmd(["git", "status", "--porcelain"], capture=True, check=False)
        if not status.stdout.strip():
            print("  ℹ️  Không có thay đổi để commit (data không đổi so với version trước)")
        else:
            run_cmd(["git", "commit", "-m", message])

        # Tạo/ghi đè tag (dùng -f để idempotent)
        run_cmd(["git", "tag", "-f", tag, "-m", message])
        print(f"  🏷️  Git tag tạo thành công: {tag}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ⚠️  git commit/tag bỏ qua (demo mode): {e}")
        return False


def dvc_push(remote: str) -> bool:
    """
    Chạy `dvc push --remote <remote>`.
    Trả về True nếu thành công.
    """
    try:
        run_cmd(["dvc", "push", "--remote", remote])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ⚠️  dvc push bỏ qua (demo mode / remote chưa cấu hình): {e}")
        return False


# ── Hàm chính ─────────────────────────────────────────────────────────────────

def version_data(
    data_dir: str,
    run_id: str,
    dvc_remote: str = DEFAULT_DVC_REMOTE,
) -> str:
    """
    Đánh version cho dataset.
    Trả về version tag đã tạo (vd: "dataset-v3").
    """
    data_path = Path(data_dir)
    registry_path = Path(VERSIONS_REGISTRY)

    print("=" * 60)
    print("📌 DATA VERSIONING")
    print(f"   data_dir  : {data_path.resolve()}")
    print(f"   run_id    : {run_id}")
    print(f"   dvc_remote: {dvc_remote}")
    print("=" * 60 + "\n")

    # ── Bước 1: Tính version tiếp theo ───────────────────────────────────────
    versions = load_versions_registry(registry_path)
    next_version_num = len(versions) + 1
    version_tag = f"dataset-v{next_version_num}"
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"📋 Bước 1/4: Version tiếp theo → {version_tag}")

    # ── Bước 2: Đếm ảnh ──────────────────────────────────────────────────────
    print("\n📋 Bước 2/4: Đếm ảnh theo class")
    counts = count_images_per_class(data_path)
    total = sum(counts.values())
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} ảnh")
    print(f"  Tổng: {total} ảnh")

    # ── Bước 3: DVC add ───────────────────────────────────────────────────────
    print("\n📋 Bước 3/4: DVC add + Git commit + Git tag")
    dvc_ok = dvc_add(data_path)
    commit_msg = f"data: snapshot {version_tag} | {total} imgs | run={run_id[:8]}"
    git_ok = git_commit_and_tag(data_path, version_tag, commit_msg)

    # ── Bước 4: DVC push ──────────────────────────────────────────────────────
    print(f"\n📋 Bước 4/4: DVC push → remote '{dvc_remote}'")
    push_ok = dvc_push(dvc_remote)

    # ── Cập nhật registry ─────────────────────────────────────────────────────
    new_entry: dict = {
        "version": version_tag,
        "version_num": next_version_num,
        "git_run_id": run_id,
        "timestamp": timestamp,
        "counts_per_class": counts,
        "total_samples": total,
        "dvc_remote": dvc_remote,
        "dvc_push_ok": push_ok,
        "git_tag_ok": git_ok,
    }
    versions.append(new_entry)
    save_versions_registry(registry_path, versions)
    print(f"\n📝 Registry cập nhật: {registry_path}")

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"✅ DATA VERSIONING DONE")
    print(f"   Version tag : {version_tag}")
    print(f"   Timestamp   : {timestamp}")
    print(f"   Total imgs  : {total}")
    print(f"   DVC push    : {'✅ OK' if push_ok else '⚠️ skipped (demo)'}")
    print(f"   Git tag     : {'✅ OK' if git_ok else '⚠️ skipped (demo)'}")
    print("=" * 60 + "\n")

    return version_tag


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh version cho dataset bằng DVC trước khi retrain"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Thư mục dataset (ImageFolder format)"
    )
    parser.add_argument(
        "--run-id", default="manual",
        help="GitHub SHA hoặc run ID để gắn vào metadata version"
    )
    parser.add_argument(
        "--dvc-remote",
        default=os.getenv("DVC_REMOTE", DEFAULT_DVC_REMOTE),
        help=(
            "Tên DVC remote để push (mặc định: localremote). "
            "Đặt DVC_REMOTE=s3remote khi deploy trên AWS."
        ),
    )
    args = parser.parse_args()

    version_tag = version_data(
        data_dir=args.data_dir,
        run_id=args.run_id,
        dvc_remote=args.dvc_remote,
    )
    print(f"VERSION_TAG={version_tag}")
