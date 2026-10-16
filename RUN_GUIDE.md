# 📖 HƯỚNG DẪN CHẠY DỰ ÁN LIBSEAT (RUN_GUIDE.md)

> **Dành cho tất cả thành viên Team 3 người (Vai A, B, C) - Vin Perception Hackathon Track 4**

Dự án **LibSeat** là hệ thống thị giác máy tính phát hiện ghế trống trong thư viện/phòng học thời gian thực, kết hợp mô hình học sâu (**Học**), làm mượt hysteresis (**Bất định**) và ma trận homography (**Hình học**).

---

## ⚡ 1. Thiết lập môi trường nhanh (Setup)

Yêu cầu: Đã cài Python 3.10, 3.11 hoặc 3.12 (khuyến nghị dùng `uv` hoặc `venv`). Không bắt buộc có GPU, hệ thống tối ưu chạy mượt trên CPU máy tính thông thường.

### Cách 1: Sử dụng công cụ siêu tốc `uv` (Khuyên dùng)
```bash
# 1. Tạo môi trường ảo
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. Cài đặt thư viện bản CPU siêu nhẹ
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

### Cách 2: Sử dụng `python -m venv` chuẩn
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 🚀 2. Hướng dẫn chạy cho từng Vai trò

### 💻 Vai A: Pipeline & Demo trực tiếp (Webcam & Video Backup)

Vai A chịu trách nhiệm duy trì hệ thống chạy mượt mà, ghi hình demo và đo FPS/độ trễ.

#### 1. Chạy với Webcam laptop (Demo chính thức)
```bash
source .venv/bin/activate
python src/app.py --source 0
```

#### 2. Chạy với Video dự phòng (Backup nếu phòng thi tối/hỏng camera)
```bash
python src/app.py --source demo_backup.mp4
```

#### ⌨️ Các phím điều khiển khi app đang mở:
- **`s` (Snapshot/Save Seats):** **Cực kỳ quan trọng!** Khi mở camera hướng vào phòng/bàn học trống, bấm `s` để chốt các vị trí ghế cố định. Sau đó người ngồi vào che khuất ghế thì hệ thống vẫn nhận diện ghế đó chuyển sang màu Đỏ (`Occupied`).
- **`r` (Reset):** Xóa bộ đệm làm mượt và bỏ chốt ghế cố định để nhận diện lại.
- **`q` (Quit):** Dừng chương trình và tự động lưu toàn bộ dữ liệu FPS, trạng thái ghế ra file `data/logs/run_log.csv`.

---

### 📐 Vai B: Kiểm thử Hình học & Làm mượt (Homography & Smoothing)

Vai B phụ trách giải thích và tinh chỉnh 2 trụ cột học thuật:
- **`src/smoothing.py` (`SeatState`):** Chống nhấp nháy chuyển màu (flickering). Tham số cần thử nghiệm:
  - `on_frames` (mặc định 10): Số frame liên tục thấy có người thì mới đổi sang màu Đỏ.
  - `off_frames` (mặc định 25): Số frame liên tục trống thì mới chuyển sang màu Xanh.
- **`src/geometry.py` (`FloorHomography`):** Biến đổi tọa độ pixel trên ảnh sang tọa độ mặt sàn (mét) bằng 4 điểm để phân biệt người đứng sau ghế và người thực sự ngồi vào ghế.

---

### 📊 Vai C: Chạy đánh giá Bằng chứng (Evaluation & Metrics)

Vai C phụ trách chuẩn bị số đo, ca lỗi và slide thuyết trình cho ban giám khảo.

#### Chạy script đánh giá trên tập dữ liệu kiểm thử Beyond:
```bash
source .venv/bin/activate
python src/eval_beyond.py
```

Kết quả in ra terminal bao gồm:
- **Tổng số ghế được kiểm tra**
- **Độ chính xác từng ghế (Accuracy %)**
- **Ma trận nhầm lẫn (Confusion Matrix: True/False Occupied)**
- Tự động xuất danh sách các ca lỗi vào file: `results/errors.csv` để đưa lên slide làm ca lỗi thực tế.

---

### ☁️ Chạy trên GPU Kaggle (video test + video demo sản phẩm)

Dùng mô hình vai C đã chọn (YOLO11s, imgsz 960, conf 0.10) trên GPU T4 của Kaggle, gán ghế và làm mượt bằng `OccupancyEngine` của vai B. Cần file `.env` ở gốc repo (đã nằm trong `.gitignore`, **không commit**) với `KAGGLE_API_TOKEN`, `NGROK_AUTHTOKEN`, `NGROK_DOMAIN`.

```bash
pip install kaggle
python kaggle/push_kaggle.py            # chạy kernel GPU trong 2 giờ (--hours để đổi)
python kaggle/push_kaggle.py status     # trạng thái kernel
python kaggle/push_kaggle.py output     # tải kết quả về kaggle/output/
```

- Token ngrok được đưa lên dataset **riêng tư** `<user>/libseat-secrets`, không nằm trong code kernel.
- Khi khởi động, kernel tự tạo video test (các video trong `video_test/`, `demo_backup.mp4` và 2 clip Pexels) và video demo sản phẩm `libseat_demo.mp4`.
- Mở `https://<NGROK_DOMAIN>`: bật webcam khi ghế còn trống (ghế tự được thêm, hoặc bấm **Thêm ngay ghế đang thấy**), **Quay video test** theo kịch bản (ghế trống → ngồi → đứng sát ghế / đi ngang → che ghế), rồi **Dựng lại video demo** để đưa bản quay của nhóm lên đầu video demo. Chi tiết: [kaggle/README.md](kaggle/README.md).
- Log `*_log.csv` theo định dạng của `libseat/src/eval_log.py`; `*_full_log.csv` giữ cả trạng thái `unknown` và cấu hình ngưỡng.
- Bấm **Tắt server** khi xong để tiết kiệm quota GPU; mọi file nằm trong Output của kernel.

---

## 📂 3. Cấu trúc thư mục dự án

```
.
├── .gitignore              # Bỏ qua virtualenv, video dung lượng lớn, checkpoints .pt
├── RUN_GUIDE.md            # Tài liệu hướng dẫn chạy này
├── requirements.txt        # Danh sách thư viện Python
├── ke_hoach_dem_cho_trong_thu_vien.md # Toàn bộ kế hoạch và rubric 100 điểm
├── demo_backup.mp4         # Video mẫu để chạy test dự phòng
├── data/
│   ├── beyond/test/        # Tập ảnh và nhãn ground-truth để đo độ chính xác
│   └── logs/               # File run_log.csv xuất ra khi chạy app.py
├── results/
│   └── errors.csv          # Danh sách ghế bị dự đoán sai phục vụ báo cáo
└── src/
    ├── app.py              # Chương trình chính (UI, Webcam/Video, Dashboard)
    ├── detect.py           # YOLO11n + ByteTrack nhận diện person & chair
    ├── geometry.py         # Tính toán IoU và ma trận biến đổi sàn Homography
    ├── smoothing.py        # Thuật toán Hysteresis làm mượt trạng thái ghế
    └── eval_beyond.py      # Đánh giá Accuracy & Confusion Matrix
```

---

## 🎯 4. Kịch bản thuyết trình 10 phút (Hackathon Presentation)

1. **Phút 0 - 1 (Vai C):** Đặt vấn đề nỗi đau sinh viên VinUni tìm chỗ học mùa thi, giải pháp LibSeat.
2. **Phút 1 - 5 (Vai A & B):** **Demo trực tiếp webcam** (hoặc `demo_backup.mp4`). Mời giám khảo tự ngồi hoặc kéo ghế. Minh họa cơ chế chống chớp nháy (Bất định) và không nhầm người đứng phía sau (Hình học).
3. **Phút 5 - 7 (Vai C):** Trình bày Bằng chứng: Độ chính xác kiểm thử trên dataset, 1 ca lỗi tiêu biểu từ `results/errors.csv` và ước tính thời gian tiết kiệm.
4. **Phút 7 - 10 (Cả nhóm):** Hỏi đáp cùng Hội đồng giám khảo.
