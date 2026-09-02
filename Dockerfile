# Dockerfile — Multi-stage build
#
# Stage 1 (builder): Cài dependencies vào virtual environment
# Stage 2 (runtime): Chỉ copy venv từ builder — không có pip, build tools
#   → Image cuối gọn hơn đáng kể

# ──────────────────────────────────────────────
# Stage 1: builder — cài deps
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Tạo virtualenv để dễ copy sang stage tiếp theo
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Cài dependencies (layer này được cache khi requirements.txt không thay đổi)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2: runtime — image production thực sự
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

WORKDIR /app

# Chỉ copy venv đã build sẵn — không cần pip hay build tools
COPY --from=builder /venv /venv

# Copy source code
COPY app/ ./app/
COPY models/ ./models/

# Tạo thư mục data/low_confidence nếu chưa tồn tại
RUN mkdir -p data/low_confidence

# Port mặc định FastAPI
EXPOSE 8000

# Health check đơn giản
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "app/server.py"]
