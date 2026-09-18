#!/bin/sh
# docker-entrypoint.sh
#
# Chạy TRƯỚC khi start node server.
# Mục đích: tạo các thư mục data/ SAU khi volume được mount (runtime),
# vì build-time mkdir bị volume mount ghi đè hoàn toàn.
#
# Vấn đề gốc:
#   Dockerfile: RUN mkdir -p data/low_confidence && chown appuser ...
#   docker-compose: volumes: - ./data:/app/data
#   → Khi container start, ./data từ host GHIBB ĐÈ /app/data trong image
#   → Các thư mục tạo lúc build biến mất → EACCES khi node cố mkdir
#
# Fix: script này tạo lại các thư mục cần thiết sau khi volume đã mount.

set -e

DATA_DIR="${LOW_CONFIDENCE_DIR%/*}"   # lấy parent của LOW_CONFIDENCE_DIR
DATA_DIR="${DATA_DIR:-/app/data}"     # fallback nếu biến chưa set

mkdir -p \
  "${LOW_CONFIDENCE_DIR:-/app/data/low_confidence}" \
  "${LABELED_DIR:-/app/data/labeled}" \
  "${RETRAIN_TRIGGER_DIR:-/app/data/retrain_trigger}" \
  "${RAW_DATA_DIR:-/app/data/raw}/Cat" \
  "${RAW_DATA_DIR:-/app/data/raw}/Dog"

echo "[entrypoint] ✅ Data directories ready"

# Chuyển quyền điều khiển sang CMD (node server.js)
exec "$@"
