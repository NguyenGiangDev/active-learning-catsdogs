"""
train/train.py
Script train ban đầu + retrain chung, nhận tham số thư mục dữ liệu.

Cách dùng:
    python train/train.py                          # dùng DATA_DIR mặc định từ config
    python train/train.py --data-dir data/raw      # chỉ định thư mục cụ thể
    python train/train.py --epochs 10              # override số epoch

Cấu trúc data_dir cần theo dạng ImageFolder:
    data/raw/
        Cat/  -> *.jpg / *.png
        Dog/  -> *.jpg / *.png
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from train.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAIN_SPLIT,
)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Trả về (train_transform, val_transform)."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def build_model(num_classes: int = 2) -> nn.Module:
    """MobileNetV2 pretrained, thay head cuối thành num_classes."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train(data_dir: str, epochs: int, save_path: str) -> None:
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, data_dir={data_dir}, epochs={epochs}")

    train_tf, val_tf = get_transforms()

    # --- Load dataset ---
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    n_train = int(len(full_dataset) * TRAIN_SPLIT)
    n_val = len (full_dataset) - n_train
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    # Áp dụng val transform cho tập val
    val_ds.dataset.transform = val_tf  # type: ignore[attr-defined]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model, loss, optimizer ---
    model = build_model(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Train loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / n_train
        train_acc = correct / n_train

        # --- Validation ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / n_val

        print(
            f"Epoch [{epoch}/{epochs}]  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        )

    # --- Lưu model ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] Model lưu tại {save_path}")

    # Lưu thêm thông tin class mapping
    class_map_path = Path(save_path).with_suffix(".classes.txt")
    class_map_path.write_text("\n".join(full_dataset.classes))
    print(f"[train] Class map lưu tại {class_map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Retrain MobileNetV2 Cats vs Dogs")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Thư mục dữ liệu (ImageFolder format)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--save-path", default=MODEL_SAVE_PATH)
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs, save_path=args.save_path)
