# train/config.py
# Các hằng số cấu hình cho quá trình train

DATA_DIR: str = "data/raw"           # thư mục dữ liệu train
MODEL_SAVE_PATH: str = "models/model_v1.pt"
IMG_SIZE: int = 224                  # kích thước ảnh đầu vào MobileNetV2
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 5                  # nhỏ — demo pipeline, không phải Kaggle
LEARNING_RATE: float = 1e-3
NUM_CLASSES: int = 2                 # 0=Cat, 1=Dog
TRAIN_SPLIT: float = 0.8            # 80% train, 20% val
RANDOM_SEED: int = 42
