# Vai C: Dữ liệu, đo, trình bày

Mọi lệnh chạy từ thư mục `libseat/`.

## 1. Dữ liệu (tối nay)
1. Tải **Beyond** (`universe.roboflow.com/lyeonsl/beyond-sub9e`) định dạng **YOLOv8**, giải nén vào `data/beyond/` (phải có `data/beyond/data.yaml`).
2. Tùy chọn: **seatanalysis** (`12dd02mm2025yyyy`) vào `data/seatanalysis/`.
3. Ghi lại giấy phép của từng bộ (xem trang dataset) vào bảng cuối file này.

## 2. Số đo trên ảnh có nhãn (0:15 – 0:45, chạy lại ở 1:15 – 1:45)
```
python src/eval_beyond.py --data data/beyond --tag beyond
python src/show_errors.py --data data/beyond --tag beyond --max 12
python src/sweep_detector.py --data data/beyond --tag beyond   # so 6 cấu hình bộ phát hiện, chọn theo CALIB
```
- Script tự đọc `data.yaml` để biết lớp nào là trống / có người (in ra để kiểm tra).
- Chia cố định CALIB/TEST theo tên file: chỉnh ngưỡng trên CALIB, **báo số trên TEST**.
- Ra: `results/metrics_beyond.md` (bảng trước/sau), `results/errors_beyond.csv`,
  ảnh ca lỗi `results/errors_beyond/` + `contact_sheet.jpg`.
- Mở `results/error_causes_beyond.csv`, điền cột `cause` (che khuất / góc nhìn / balo / ánh sáng / nhãn sai), chọn **1 ca tiêu biểu** cho slide.

**Lưu ý trung thực:** luật "điểm chân" trên ảnh tĩnh chỉ là bản gần đúng của homography (dataset không có 4 mốc sàn). Homography thật được kiểm chứng trên demo webcam.

## 3. Số đo trên demo (1:45 – 2:15)
Thống nhất với vai A định dạng log: `t,frame,seat_id,raw_occ,smooth_occ` (1 dòng / ghế / khung).
Bấm giờ ngồi xuống → đổi màu 5–10 lần, ghi vào `data/logs/latency.csv` (`trial,latency_s`).
```
python src/eval_log.py --log data/logs/run1.csv --stopwatch data/logs/latency.csv
```
Ra: FPS, số lần nhấp nháy thô → sau làm mượt, độ trễ.

## 4. Impact
Khảo sát 10–15 bạn, ghi `data/survey.csv` (`minutes_search,visits_per_day,would_use`).
```
python src/impact.py --survey data/survey.csv --users 500 --saved-share 0.6
```
`--users` và `--saved-share` là **giả định**, phải nói rõ khi trình bày.

Câu hỏi khảo sát:
1. Mỗi lần vào thư viện, bạn mất khoảng bao nhiêu phút để tìm được chỗ ngồi? (số phút)
2. Những ngày có đến thư viện, bạn vào mấy lần? (số lần)
3. Nếu có màn hình/app báo ghế trống, bạn có xem trước khi đi không? (Có/Không)

## 5. Video dự phòng
Quay 60–90 s từ chính app demo (thành viên đóng vai): phòng trống → 1 người ngồi → người đi ngang không ngồi → che ghế (ca lỗi). Lưu `demo_backup.mp4`, **không công khai** (đã có trong `.gitignore`).

## 6. Giấy phép dataset (điền sau khi kiểm tra)
| Dataset | Giấy phép | Link | Đã kiểm tra |
|---|---|---|---|
| Beyond v3 (233 ảnh, resize 960×960 stretch) | CC BY 4.0 | universe.roboflow.com/lyeonsl/beyond-sub9e | [x] 2026-10-16 |
| 12dd02mm2025yyyy v1 (114 ảnh, CCTV văn phòng) | CC BY 4.0 | universe.roboflow.com/seatanalysis/12dd02mm2025yyyy | [x] 2026-10-16 |

Ghi chú dữ liệu: Beyond chỉ 12% ghế có người (độ chính xác thường bị thổi phồng, báo kèm balanced accuracy và recall ghế có người); một số ảnh Beyond bị xoay 90°; seatanalysis là các khung của cùng một camera nên CALIB/TEST cùng cảnh.
