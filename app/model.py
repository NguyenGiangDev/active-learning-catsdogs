"""
app/model.py
Load trọng số MobileNetV2 một lần lúc khởi động, expose hàm predict().
"""

from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# ── Hằng số cấu hình ─────────────────────────────────────────────────────────
MODEL_PATH: str = "models/model_v1.pt"
CLASS_MAP_PATH: str = "models/model_v1.classes.txt"
IMG_SIZE: int = 224
DEFAULT_CLASSES: list[str] = ["Cat", "Dog"]  # fallback nếu chưa có .classes.txt

# ── Transform inference (giống val_transform trong train.py) ──────────────────
_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _load_classes(path: str) -> list[str]:
    p = Path(path)
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return DEFAULT_CLASSES


def _build_model(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


class _ModelWrapper:
    """Singleton wrapper giữ model trong bộ nhớ."""

    _instance: ClassVar["_ModelWrapper | None"] = None

    def __init__(self) -> None:
        self.classes: list[str] = _load_classes(CLASS_MAP_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model(len(self.classes)).to(self.device)

        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[model] Loaded {MODEL_PATH} on {self.device}, classes={self.classes}")

    @classmethod
    def get(cls) -> "_ModelWrapper":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Nhận PIL.Image, trả về (label, confidence).
        confidence ∈ [0.0, 1.0].
        """
        tensor = _INFER_TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])


def predict(image: Image.Image) -> tuple[str, float]:
    """API công khai — dùng singleton bên trong."""
    return _ModelWrapper.get().predict(image)
