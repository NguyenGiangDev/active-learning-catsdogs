# Active Learning — Phân loại Chó/Mèo

Pipeline MLOps nhỏ minh họa vòng lặp **Active Learning** với Human-in-the-loop:
ảnh có confidence thấp được đưa sang giao diện gán nhãn, sau đó tự động trigger retrain qua GitHub Actions.

---

## Kiến trúc tổng quan

```
Ảnh upload
    │
    ▼
FastAPI /predict (MobileNetV2)
    │
    ├─ confidence > 80% ──────────────────────► Trả kết quả ngay
    │
    └─ confidence ≤ 80% ──► data/low_confidence/
                                    │
                                    ▼
                            Label Web UI (port 4000)
                            Human gán nhãn Cat / Dog
                                    │
                                    ▼
                            data/labeled/
                                    │
                            (đủ RETRAIN_THRESHOLD)
                                    │
                                    ▼
                        GitHub Actions — job retrain
                        python train/train.py
                                    │
                                    ▼
                        models/model_v1.pt (mới)
                        ► Docker build & push
                        ► Deploy tự động (self-hosted runner)
```

---

## Cấu trúc thư mục thực tế

```
active-learning-catsdogs/
├── app/
│   ├── main.py          # FastAPI: POST /predict, GET /health
│   ├── model.py         # Load MobileNetV2, hàm predict(image) → (label, confidence)
│   ├── filter.py        # is_uncertain(confidence, threshold) → bool
│   ├── validate.py      # Kiểm tra JPEG/PNG hợp lệ trước khi inference
│   └── server.py        # Entrypoint: python -m app.server (dùng bởi Dockerfile)
├── train/
│   ├── train.py         # Script train + retrain, nhận --data-dir / --epochs
│   └── config.py        # Hằng số: IMG_SIZE=224, BATCH_SIZE=32, NUM_EPOCHS=5, ...
├── label-web/           # Service gán nhãn (Node.js / Express)
│   ├── server.js        # REST API + serve giao diện gán nhãn
│   ├── public/          # Frontend HTML/CSS/JS
│   └── Dockerfile       # Image riêng cho label-web
├── monitoring/
│   └── drift_check.py   # Đếm tỷ lệ low-confidence, gửi Slack nếu vượt ngưỡng
├── tests/
│   ├── test_filter.py   # Unit test bộ lọc confidence
│   └── test_validate.py # Unit test validate ảnh
├── data/
│   ├── raw/             # Dataset train (gitignore, DVC-tracked)
│   ├── low_confidence/  # Ảnh cần gán nhãn (generated runtime)
│   ├── labeled/         # Kết quả gán nhãn (JSON)
│   └── retrain_trigger/ # File JSON trigger retrain
├── models/              # Trọng số model (DVC-tracked, gitignored)
├── .github/workflows/
│   └── cicd.yml         # CI/CD/CT: test → build & push → retrain → deploy
├── Dockerfile           # Multi-stage build cho FastAPI service
├── docker-compose.yml   # Hai service: api (8000) + label-web (4000)
├── requirements.txt     # Python deps (FastAPI, PyTorch, DVC, ...)
├── requirements-test.txt
└── .dvc/                # DVC config (sau dvc init)
```

---

## Yêu cầu

| Thành phần | Phiên bản |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ (cho label-web) |
| Docker + Docker Compose | v2+ |
| DVC | >= 3.51 |

---

## Chạy local (không Docker)

### 1. Cài dependencies Python

```bash
pip install -r requirements.txt
```

### 2. Tải dataset (Cats vs Dogs)

```bash
python download_data.py
```

Dataset sẽ được lưu vào `data/raw/` theo cấu trúc `ImageFolder` (`Cat/`, `Dog/`).

### 3. Train model

```bash
python train/train.py --data-dir data/raw --epochs 5
# Trọng số lưu tại: models/model_v1.pt
```

### 4. Chạy FastAPI

```bash
uvicorn app.main:app --reload --port 8000
```

Thử với curl:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/cat.jpg"
# {"label":"Cat","confidence":0.9741,"uncertain":false,"saved_for_review":false}
```

### 5. Chạy Label Web (gán nhãn)

```bash
cd label-web
npm install
node server.js
# Giao diện gán nhãn: http://localhost:4000
```

---

## Chạy bằng Docker Compose

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| FastAPI classifier | http://localhost:8000 |
| Label Web UI | http://localhost:4000 |
| API docs (Swagger) | http://localhost:8000/docs |

Cả hai service dùng chung volume `./data` để trao đổi ảnh low-confidence.

### Biến môi trường (tùy chọn)

Tạo file `.env` ở thư mục gốc:

```env
# Ngưỡng confidence để đưa ảnh vào review (mặc định 0.80)
CONFIDENCE_THRESHOLD=0.80

# Số ảnh label tối thiểu để trigger retrain (mặc định 10)
RETRAIN_THRESHOLD=10

# Để label-web tự động trigger GitHub Actions khi đủ ngưỡng:
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=active-learning-catsdogs
```

---

## Vòng lặp Active Learning

1. **Upload ảnh** → `POST /predict`
2. Nếu `confidence ≤ 0.80` → ảnh lưu vào `data/low_confidence/`
3. Mở **Label Web** (`http://localhost:4000`) → gán nhãn Cat / Dog
4. Khi đủ `RETRAIN_THRESHOLD` ảnh được gán nhãn:
   - File `data/retrain_trigger/trigger_{timestamp}.json` được tạo
   - Nếu `GITHUB_TOKEN` đã cấu hình → gọi GitHub `repository_dispatch` event `retrain-requested`
   - GitHub Actions chạy job `retrain` → `python train/train.py`
5. Sau khi retrain: job `build-and-push` build image mới, job `deploy` pull và restart container

---

## CI/CD/CT — GitHub Actions

File: [`.github/workflows/cicd.yml`](.github/workflows/cicd.yml)

| Job | Trigger | Runner | Mô tả |
|---|---|---|---|
| `test` | push / PR vào `main` | self-hosted | Chạy `pytest tests/` |
| `build-and-push` | push vào `main` (sau khi test pass) | ubuntu-latest | Build 2 Docker image (api + label-web), push lên Docker Hub |
| `retrain` | `workflow_dispatch` thủ công hoặc `repository_dispatch` từ label-web | ubuntu-latest | `python train/train.py --epochs 5` |
| `deploy` | Sau khi `build-and-push` thành công | self-hosted (WSL) | `docker compose up -d --force-recreate` |

### GitHub Secrets cần thêm

Vào **Settings → Secrets and variables → Actions** của repo:

| Secret | Mô tả |
|---|---|
| `DOCKERHUB_USERNAME` | Tên user Docker Hub |
| `DOCKERHUB_TOKEN` | Access token Docker Hub |

> **Lưu ý**: `GITHUB_TOKEN` cho label-web là Personal Access Token riêng (không phải `secrets.GITHUB_TOKEN` của Actions).
> Cần thêm vào file `.env` khi deploy thật.

---

## Monitoring drift

```bash
# Kiểm tra 24h gần đây (mặc định)
python monitoring/drift_check.py

# Kiểm tra 6h gần đây
python monitoring/drift_check.py --hours 6

# Dry-run — in ra console, không gửi Slack
python monitoring/drift_check.py --dry-run
```

Biến môi trường:

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
DRIFT_THRESHOLD=0.40   # cảnh báo khi > 40% ảnh bất định
```

---

## Quản lý dữ liệu với DVC

```bash
# Khởi tạo (chỉ làm 1 lần)
dvc init
dvc remote add -d localremote ../catdog-dvc-storage

# Version hóa dataset và model
dvc add data/raw
dvc add models
git add data/raw.dvc models.dvc .gitignore
git commit -m "dataset v1.0"
dvc push

# Pattern lặp lại sau mỗi vòng gán nhãn + retrain:
dvc add data/raw && git commit -am "dataset vX.Y" && dvc push
```

---

## Chạy tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```

| File test | Bao phủ |
|---|---|
| `tests/test_filter.py` | `is_uncertain()` — bao gồm trường hợp biên |
| `tests/test_validate.py` | `validate_image()` — JPEG/PNG hợp lệ, file giả, quá kích thước |

---

## Stack công nghệ

| Lớp | Công nghệ |
|---|---|
| ML serving | FastAPI + Uvicorn |
| Model | MobileNetV2 pretrained (PyTorch / torchvision) |
| Label UI | Node.js / Express + Vanilla JS |
| Containerization | Docker multi-stage + Docker Compose |
| CI/CD/CT | GitHub Actions (4 jobs) |
| Data versioning | DVC |
| Monitoring | Python script + Slack Webhook |
| Testing | pytest + pytest-asyncio + httpx |
