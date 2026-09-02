"""
app/main.py
FastAPI app — endpoint /predict phân loại ảnh Chó/Mèo với Active Learning loop.

Khi model bất định (confidence ≤ threshold), ảnh + metadata JSON được lưu
vào data/low_confidence/ để con người gán nhãn lại qua Label Studio.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.filter import is_uncertain
from app.model import predict
from app.validate import ImageValidationError, validate_image

# ── Hằng số cấu hình ─────────────────────────────────────────────────────────
LOW_CONFIDENCE_DIR: str = os.getenv("LOW_CONFIDENCE_DIR", "data/low_confidence")
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="FastAPI service phân loại Chó/Mèo với vòng lặp Active Learning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    """Tạo thư mục cần thiết khi khởi động."""
    Path(LOW_CONFIDENCE_DIR).mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check — dùng cho Docker HEALTHCHECK và GitHub Actions."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Phân loại ảnh upload.

    Returns:
        {
            "label": "Cat" | "Dog",
            "confidence": 0.95,
            "uncertain": false,
            "saved_for_review": false
        }
    """
    # 1. Đọc bytes
    data = await file.read()

    # 2. Validate
    try:
        image = validate_image(
            data,
            filename=file.filename or "",
            content_type=file.content_type or "",
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 3. Inference
    label, confidence = predict(image)

    # 4. Kiểm tra độ bất định
    uncertain = is_uncertain(confidence, threshold=CONFIDENCE_THRESHOLD)
    saved_for_review = False

    if uncertain:
        saved_for_review = _save_for_review(data, file.filename or "upload.jpg", label, confidence)

    return JSONResponse(
        {
            "label": label,
            "confidence": round(confidence, 4),
            "uncertain": uncertain,
            "saved_for_review": saved_for_review,
        }
    )


def _save_for_review(
    data: bytes,
    original_filename: str,
    predicted_label: str,
    confidence: float,
) -> bool:
    """
    Lưu ảnh + metadata JSON vào LOW_CONFIDENCE_DIR.
    Trả về True nếu lưu thành công, False nếu có lỗi.
    """
    try:
        file_id = uuid.uuid4().hex
        ext = Path(original_filename).suffix.lower() or ".jpg"
        base_path = Path(LOW_CONFIDENCE_DIR) / file_id

        # Lưu ảnh
        img_path = base_path.with_suffix(ext)
        img_path.write_bytes(data)

        # Lưu metadata
        meta = {
            "id": file_id,
            "original_filename": original_filename,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = base_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        return True
    except Exception as exc:  # noqa: BLE001
        # Không raise — predict vẫn trả kết quả dù lưu file thất bại
        print(f"[main] Lỗi khi lưu ảnh bất định: {exc}")
        return False
