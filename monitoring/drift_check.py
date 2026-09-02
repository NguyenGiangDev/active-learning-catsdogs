"""
monitoring/drift_check.py
Giám sát drift đơn giản: đếm tỷ lệ ảnh low-confidence trong khoảng thời gian
gần đây so với tổng số dự đoán (đọc từ file metadata JSON trong low_confidence/).

Nếu tỷ lệ vượt ngưỡng DRIFT_THRESHOLD, gửi cảnh báo đến Slack Webhook.

Cách dùng:
    python monitoring/drift_check.py               # kiểm tra 24h gần đây
    python monitoring/drift_check.py --hours 6     # kiểm tra 6h gần đây
    python monitoring/drift_check.py --dry-run     # in ra console, không gửi Slack

Biến môi trường cần thiết:
    SLACK_WEBHOOK_URL — URL Incoming Webhook của Slack (không hardcode).
    DRIFT_THRESHOLD   — Ngưỡng tỷ lệ bất định (mặc định 0.40 = 40%).
    LOW_CONFIDENCE_DIR — Thư mục chứa metadata JSON (mặc định data/low_confidence).
    TOTAL_PREDICTIONS_LOG — File log tổng số dự đoán (tùy chọn, xem bên dưới).

Ghi chú về TOTAL_PREDICTIONS_LOG:
    Vì FastAPI không persist bộ đếm tổng số predict theo mặc định, drift_check.py
    dùng số metadata JSON trong thư mục làm đại diện cho "ảnh bất định trong kỳ".
    Để tính tỷ lệ chính xác hơn, thêm middleware ghi log vào app/main.py và
    trỏ TOTAL_PREDICTIONS_LOG vào file đó.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests  # dùng để gửi Slack webhook

# ── Hằng số cấu hình (đọc từ biến môi trường) ────────────────────────────────
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.40"))
LOW_CONFIDENCE_DIR: str = os.getenv("LOW_CONFIDENCE_DIR", "data/low_confidence")
TOTAL_PREDICTIONS_LOG: str = os.getenv("TOTAL_PREDICTIONS_LOG", "")


def count_low_confidence_in_window(directory: str, hours: int) -> tuple[int, int]:
    """
    Đọc các file .json trong directory, đếm số ảnh bất định
    trong khoảng `hours` giờ gần đây.

    Returns:
        (count_in_window, total_json_files)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    dir_path = Path(directory)

    if not dir_path.exists():
        return 0, 0

    json_files = list(dir_path.glob("*.json"))
    total = len(json_files)
    count_in_window = 0

    for jf in json_files:
        try:
            meta = json.loads(jf.read_text())
            ts_str = meta.get("timestamp", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    count_in_window += 1
        except Exception:  # noqa: BLE001
            # Bỏ qua file lỗi
            continue

    return count_in_window, total


def read_total_predictions(log_path: str) -> int:
    """
    Đọc tổng số predict từ file log (mỗi dòng = 1 dự đoán).
    Trả về 0 nếu file không tồn tại.
    """
    p = Path(log_path)
    if not p.exists():
        return 0
    try:
        return sum(1 for line in p.read_text().splitlines() if line.strip())
    except Exception:  # noqa: BLE001
        return 0


def send_slack_alert(webhook_url: str, message: str) -> bool:
    """
    Gửi tin nhắn đến Slack Incoming Webhook.
    Trả về True nếu thành công, False nếu thất bại.
    """
    try:
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[drift_check] Lỗi gửi Slack: {exc}", file=sys.stderr)
        return False


def run_drift_check(hours: int = 24, dry_run: bool = False) -> dict:
    """
    Chạy kiểm tra drift và gửi cảnh báo nếu cần.

    Returns:
        dict chứa kết quả kiểm tra để dễ test/log.
    """
    low_conf_count, total_json = count_low_confidence_in_window(LOW_CONFIDENCE_DIR, hours)

    # Tổng số dự đoán: ưu tiên từ log file nếu có, fallback về total_json
    if TOTAL_PREDICTIONS_LOG:
        total_predictions = read_total_predictions(TOTAL_PREDICTIONS_LOG)
    else:
        # Giả định: số metadata JSON ≈ tổng số predict có ảnh bất định.
        # Tỷ lệ này sẽ bị sai lệch nếu không có log. Ghi chú rõ.
        total_predictions = total_json
        if total_predictions == 0:
            print("[drift_check] Không có dữ liệu predict. Bỏ qua.")
            return {"low_conf_count": 0, "total": 0, "ratio": 0.0, "alert_sent": False}

    ratio = low_conf_count / total_predictions if total_predictions > 0 else 0.0
    alert_needed = ratio > DRIFT_THRESHOLD

    result = {
        "period_hours": hours,
        "low_conf_in_window": low_conf_count,
        "total_predictions": total_predictions,
        "ratio": round(ratio, 4),
        "threshold": DRIFT_THRESHOLD,
        "alert_needed": alert_needed,
        "alert_sent": False,
    }

    status_icon = "🔴" if alert_needed else "🟢"
    print(
        f"[drift_check] {status_icon} "
        f"low_conf={low_conf_count}/{total_predictions} "
        f"ratio={ratio:.1%} threshold={DRIFT_THRESHOLD:.0%}"
    )

    if alert_needed:
        message = (
            f"⚠️ *Drift cảnh báo* [{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n"
            f"• Tỷ lệ ảnh bất định: *{ratio:.1%}* (ngưỡng: {DRIFT_THRESHOLD:.0%})\n"
            f"• Ảnh bất định trong {hours}h: {low_conf_count}/{total_predictions}\n"
            f"• Thư mục: `{LOW_CONFIDENCE_DIR}`\n"
            f"→ Kiểm tra Label Studio và xem xét retrain."
        )
        if dry_run:
            print(f"[drift_check] DRY-RUN — Sẽ gửi Slack:\n{message}")
            result["alert_sent"] = False
        elif SLACK_WEBHOOK_URL:
            result["alert_sent"] = send_slack_alert(SLACK_WEBHOOK_URL, message)
        else:
            print(
                "[drift_check] SLACK_WEBHOOK_URL chưa được đặt. "
                "Đặt biến môi trường này để bật cảnh báo Slack.",
                file=sys.stderr,
            )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift check cho Cats vs Dogs classifier")
    parser.add_argument(
        "--hours", type=int, default=24, help="Cửa sổ thời gian kiểm tra (giờ, mặc định 24)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In kết quả ra console thay vì gửi Slack",
    )
    args = parser.parse_args()

    result = run_drift_check(hours=args.hours, dry_run=args.dry_run)
    sys.exit(0)
