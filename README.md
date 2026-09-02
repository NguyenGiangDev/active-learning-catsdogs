---
name: setup-catdog-active-learning
description: Dựng khung và triển khai dự án MLOps Active Learning cho bài toán phân loại Chó/Mèo — dịch vụ FastAPI phân loại ảnh (MobileNetV2) kèm vòng lặp Active Learning dựa trên độ tự tin, dán nhãn Human-in-the-loop bằng Label Studio, quản lý phiên bản dữ liệu bằng DVC, pipeline CI/CD/CT bằng GitHub Actions, và giám sát drift đơn giản. Dùng skill này bất cứ khi nào người dùng yêu cầu thiết lập, dựng khung, xây dựng, hoặc tiếp tục dự án cụ thể này — kể cả các yêu cầu như "set up FastAPI service", "viết hàm lọc confidence", "thêm Label Studio", "cấu hình DVC", "viết workflow CI/CD", "thêm monitoring drift", hoặc "docker-compose cho project này" — kể cả khi họ không nhắc tên dự án. Đây là dự án cá nhân/portfolio quy mô nhỏ: ưu tiên code nhỏ gọn, chạy được, có chú thích rõ ràng hơn là làm production-grade. Giữ số lượng dependency và bước cài đặt ở mức tối thiểu.
---

# Thiết lập: Dự án Active Learning Phân loại Chó/Mèo

## Dự án này là gì

Một pipeline MLOps nhỏ minh họa Active Learning (Học chủ động):

```
Ảnh từ người dùng → FastAPI (MobileNetV2) → chắc chắn? → trả kết quả
                                          → không chắc (confidence ≤ 80%)? → data/low_confidence/
                                               → Label Studio (con người gán lại nhãn)
                                               → DVC version hóa dataset mới
                                               → GitHub Actions retrain + rebuild Docker image
                                               → kiểm tra drift đơn giản, báo Slack nếu tỷ lệ lỗi tăng đột biến
```

Hãy coi đây là dự án cá nhân/portfolio chill, không phải sản phẩm doanh nghiệp:
- Dùng tập con dữ liệu nhỏ (vài nghìn ảnh, vài epoch) là ổn.
- DVC remote local (một thư mục thứ hai trên máy) là đủ — không cần S3/GCS trừ khi được yêu cầu.
- Một workflow GitHub Actions đơn giản là đủ — không cần nhiều job phức tạp.
- Trigger thủ công/giả lập là chấp nhận được ở bất kỳ đâu mà webhook "thật" quá phức tạp (ghi chú rõ trong code comment và nói với người dùng).
- Ưu tiên làm từng phần *chạy được từ đầu đến cuối* hơn là đánh bóng một phần nào đó quá kỹ.

## Trước khi viết bất kỳ đoạn code nào

1. Kiểm tra những gì đã có sẵn: `view` thư mục dự án. Nếu `app/`, `docker-compose.yml`, v.v. đã tồn tại, đọc trước khi thêm/sửa — đừng ghi đè âm thầm.
2. Chỉ hỏi người dùng (ngắn gọn, từng câu một, không phải danh sách câu hỏi) khi thực sự cần thiết — ví dụ: "bạn đã tải sẵn dataset chưa, hay mình viết luôn bước tải về?". Còn lại thì tự chọn phương án hợp lý và nói rõ mình đã giả định gì.
3. Xác định người dùng đang cần giai đoạn nào (xem phần Các giai đoạn bên dưới) và làm tốt giai đoạn đó, thay vì cố tạo toàn bộ dự án trong một lần trừ khi được yêu cầu rõ ràng.

## Cấu trúc thư mục mục tiêu

```
project-root/
├── app/
│   ├── main.py              # FastAPI app: endpoint /predict
│   ├── model.py              # load model + wrapper inference
│   ├── filter.py              # bộ lọc độ bất định (confidence)
│   └── validate.py            # validate ảnh (định dạng, decode an toàn)
├── train/
│   ├── train.py               # script train ban đầu + retrain (dùng chung, nhận tham số thư mục data)
│   └── config.py
├── data/
│   ├── raw/                   # tập train gốc (gitignore, dvc-tracked)
│   └── low_confidence/        # ảnh bị lọc do độ tự tin thấp
├── models/                    # trọng số đã lưu (dvc-tracked, không commit trực tiếp vào git)
├── monitoring/
│   └── drift_check.py         # đếm tỷ lệ low_confidence, gửi Slack webhook
├── tests/
│   ├── test_filter.py
│   └── test_validate.py
├── .github/workflows/
│   └── cicd.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .dvc/ (sau khi `dvc init`)
└── README.md
```

## Các giai đoạn (thực hiện theo thứ tự này nếu làm từ đầu)

### Giai đoạn 1 — Model + FastAPI serving
- `train/train.py`: transfer learning MobileNetV2 (pretrained từ torchvision), train trên tập con Cats-vs-Dogs, lưu trọng số vào `models/model_v1.pt`. Mặc định để số epoch nhỏ (3–5) — đây là pipeline demo, không phải bài dự thi Kaggle.
- `app/model.py`: load trọng số một lần lúc khởi động, expose hàm `predict(image) -> (label, confidence)`.
- `app/validate.py`: kiểm tra file upload là JPEG/PNG thật, decode được, dưới giới hạn kích thước hợp lý trước khi đưa vào model — đây là hàm mà `tests/test_validate.py` cần bao phủ.
- `app/main.py`: FastAPI app, `async def predict(file: UploadFile)`, trả về `{"label": ..., "confidence": ...}`. Gắn bộ lọc (Giai đoạn 2) vào trước khi trả kết quả.
- Chạy thử local bằng `uvicorn app.main:app --reload --port 8000` và xác nhận bằng curl/Postman.

### Giai đoạn 2 — Bộ lọc độ bất định (lõi của Active Learning)
- `app/filter.py`: hàm thuần `is_uncertain(confidence: float, threshold: float = 0.8) -> bool`. Giữ hàm này pure/testable — không có I/O bên trong.
- Trong `app/main.py`, khi `is_uncertain(...)` trả về true, lưu ảnh + một file metadata JSON nhỏ đi kèm (timestamp, nhãn dự đoán, confidence) vào `data/low_confidence/`.
- `tests/test_filter.py`: bao phủ trường hợp biên (confidence đúng bằng threshold) và cả hai phía của ngưỡng — đây là test mà CI sẽ chạy.

### Giai đoạn 3 — Docker Compose + Label Studio
- `Dockerfile` cho app FastAPI (base image Python slim, cài `requirements.txt`, copy `app/` và `models/`).
- `docker-compose.yml`: hai service — `api` (build từ Dockerfile, port 8000) và `label-studio` (image chính thức `heartexlabs/label-studio`, port 8080) — dùng chung volume `./data` để Label Studio đọc được `low_confidence/`.
- Sau khi chạy được, hướng dẫn người dùng cách trỏ Local Storage của Label Studio vào `/label-studio/data/low_confidence` và thiết lập giao diện gán nhãn nhị phân Cat/Dog đơn giản — phần này là cấu hình UI thủ công, không nên cố script hóa mù quáng.

### Giai đoạn 4 — DVC
- `dvc init`, sau đó `dvc remote add -d localremote ../catdog-dvc-storage` (một thư mục anh em trên máy là đủ cho dự án cá nhân).
- `dvc add data/raw` và `dvc add models` — commit các file `.dvc` sinh ra vào git, gitignore dữ liệu/trọng số thật.
- Cho người dùng thấy pattern 2 lệnh họ sẽ lặp lại sau mỗi vòng gán nhãn: `dvc add data/raw && git commit -am "dataset vX.Y" && dvc push`.

### Giai đoạn 5 — GitHub Actions CI/CD/CT
- `.github/workflows/cicd.yml`:
  - Job `test`: khi push/PR, `pip install -r requirements.txt`, chạy `pytest`.
  - Job `build-and-push`: cần job `test` pass trước, `docker build`, push lên Docker Hub (dùng `${{ secrets.DOCKERHUB_TOKEN }}` — nhắc người dùng thêm secret này, không bao giờ hardcode credentials).
  - Job `retrain` (tùy chọn) kích hoạt bằng `workflow_dispatch` hoặc `repository_dispatch` (từ webhook Label Studio) — nếu người dùng chưa cấu hình webhook thật, dùng `workflow_dispatch` (nút bấm thủ công) và ghi chú trong comment rằng có thể đổi sang `repository_dispatch` sau.
- Chỉ nên có 2–3 job như trên. Đừng tự thêm job khác (lint, coverage, matrix nhiều OS) trừ khi được yêu cầu.

### Giai đoạn 6 — Monitoring
- `monitoring/drift_check.py`: đọc log dự đoán gần đây hoặc đếm số file trong `data/low_confidence/` theo khoảng thời gian, tính tỷ lệ so với tổng số dự đoán, gửi tin nhắn văn bản đơn giản đến Slack Incoming Webhook URL (đọc từ biến môi trường, không hardcode) nếu tỷ lệ vượt 40%.
- Với dự án cá nhân, có thể chạy dạng cron/script thủ công — không cần dựng scheduler service trừ khi được yêu cầu.

## Quy ước áp dụng xuyên suốt

- Python 3.10+, có type hint trên chữ ký hàm.
- Các giá trị cấu hình (threshold, port, path) đặt thành hằng số ở đầu file liên quan hoặc trong `config.py` nhỏ gọn — không rải số ma thuật (magic number) khắp nơi.
- Mọi secret bên ngoài (Docker Hub token, Slack webhook URL, AWS keys) đều lấy từ biến môi trường hoặc GitHub Actions secret — không bao giờ commit vào code.
- Sau khi tạo file cho mỗi giai đoạn, nói rõ lệnh chính xác để người dùng chạy thử và xem kết quả (`uvicorn ...`, `docker compose up`, `pytest`, `dvc status`, v.v.) thay vì để họ tự đoán.
- Dùng `create_file` cho file mới và `str_replace` để sửa các file đã tạo trước đó trong cuộc trò chuyện — đừng viết lại toàn bộ file đã tạo trừ khi người dùng muốn viết lại từ đầu.

## Khi người dùng yêu cầu "làm hết một lần"

Vẫn dựng từng giai đoạn qua các lần gọi tool riêng biệt (để mỗi phần dễ đọc và test độc lập), nhưng không cần dừng lại xin xác nhận giữa các giai đoạn — chỉ cần tường thuật ngắn gọn ("đang setup FastAPI serving, tiếp theo là bộ lọc, rồi đến compose...") và tiếp tục làm.
