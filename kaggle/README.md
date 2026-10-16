# LibSeat trên GPU Kaggle

Bản tích hợp cuối của LibSeat. Chạy trên GPU T4 miễn phí của Kaggle, mở cho laptop qua ngrok:

- **Demo webcam trên trình duyệt**: khung webcam gửi lên GPU, nhận lại khung đã vẽ và bảng đếm.
- **Video test**: tự xử lý các video trong `video_test/`, `demo_backup.mp4` và 2 clip Pexels khi khởi động.
- **Video demo sản phẩm** `libseat_demo.mp4`: thẻ giới thiệu, pipeline, các clip đã chạy, số đo của vai C.

Mô hình đúng cấu hình vai C đã đo: **YOLO11s, 960 px, conf 0,10**, cộng YOLO11s-pose để phân biệt ngồi/đứng. Gán người vào ghế (`assign_people`) và làm mượt (`SeatState`) dùng code của vai B trong `src/`.

## Cần có

1. `pip install kaggle` (đã thử với kaggle 2.2.4).
2. File `.env` ở gốc repo. File này đã nằm trong `.gitignore`, **không commit**:
   ```
   KAGGLE_API_TOKEN=...      # kaggle.com/settings/api → Generate New Token
   NGROK_AUTHTOKEN=...       # dashboard.ngrok.com
   NGROK_DOMAIN=ten-ban.ngrok-free.app   # domain tĩnh miễn phí của ngrok
   ```
3. Tài khoản Kaggle đã xác minh số điện thoại (kernel cần GPU và Internet).

## Lệnh

```bash
python kaggle/push_kaggle.py                 # gửi token + video_test/, chạy kernel 2 giờ
python kaggle/push_kaggle.py --hours 1       # đổi thời gian giữ server (tối đa 11 giờ)
python kaggle/push_kaggle.py --keep-videos   # dùng lại video đã gửi lần trước, không tải lại
python kaggle/push_kaggle.py --videos ""     # không gửi video_test/
python kaggle/push_kaggle.py status          # trạng thái kernel
python kaggle/push_kaggle.py output          # tải kết quả về kaggle/output/
```

`push_kaggle.py` tạo 2 dataset **riêng tư** trên tài khoản Kaggle của bạn: `libseat-secrets` chứa token ngrok (để token không nằm trong code kernel) và `libseat-videos` chứa `video_test/` (tên file được đổi sang không dấu). Kernel `libseat-gpu-demo` là kernel riêng tư.

## Dùng trang web

Mở `https://<NGROK_DOMAIN>` sau khoảng 1–3 phút (ngrok có thể hỏi xác nhận lần đầu).

1. **Bật webcam** khi ghế còn trống. Ghế được tự thêm sau khi thấy liên tục 0,5 s, giữ ID kể cả khi bị che. Nút **Thêm ngay ghế đang thấy** để không phải chờ.
2. Người ngồi yên 1,5 s ở chỗ chưa có ghế tạo ghế suy ra (`S5*`). Người đang đứng không chiếm ghế.
3. **Quay video test**, diễn: ghế trống → ngồi → đứng sát ghế / đi ngang → che ghế. Bấm lần nữa để lưu.
4. **Dựng lại video demo**: đưa bản quay mới nhất lên đầu video demo.
5. **Xử lý file video**: tải một video lên để chạy trên GPU.
6. **Tắt server** khi xong để đỡ tốn quota GPU. Mọi file nằm trong Output của kernel.

## Kết quả (`kaggle/output/`)

| File | Nội dung |
|---|---|
| `test_<tên>.mp4` | Video đã vẽ: ghế xanh/đỏ/xám, người kèm ID và tư thế, bảng đếm |
| `<tên>_log.csv` | `t,frame,seat_id,raw_occ,smooth_occ`, đọc được bằng `libseat/src/eval_log.py` |
| `<tên>_full_log.csv` | Có thêm trạng thái `unknown`, ghế ngoài khung, ID người, nguồn ghế, cấu hình ngưỡng |
| `<tên>_summary.json` | Số ghế, số ID người, nhấp nháy thô → sau làm mượt, lượt người đứng bị loại, ms/khung |
| `libseat_demo.mp4` | Video demo sản phẩm |

## Tham số chính (đầu file `libseat_kaggle.py`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `CONF` | 0,10 | Ngưỡng phát hiện, vai C chọn trên CALIB |
| `NEW_TRACK_CONF` / `CONFIRM_FRAMES` | 0,35 / 3 | Người mới phải có hộp ≥ 0,35 và xuất hiện 3 khung mới được tính |
| `TRACK_SECONDS` | 3 s | Giữ ID người khi bị che |
| `SEAT_CONF` / `SEAT_NEW_SECONDS` | 0,35 / 0,5 s | Hộp chair thành ghế mới khi đủ tin cậy và thấy liên tục |
| `SIT_INFER_SECONDS` | 1,5 s | Ngồi yên ở chỗ chưa có ghế thì suy ra ghế |
| `ON_S` / `OFF_S` | 0,5 s / 1,0 s | Làm mượt của vai B |
| `GAP_VIDEO` / `GAP_WEBCAM` | 0,5 s / 2,5 s | Webcam qua ngrok mỗi khung mất 1,1–1,5 s nên nới có chủ đích |

## Lưu ý

- Mỗi tài khoản ngrok miễn phí chỉ giữ được một phiên. Tắt kernel cũ (nút **Tắt server**) trước khi đẩy bản mới, nếu không domain sẽ bị trùng.
- Trong lúc server chạy, ai có link ngrok cũng mở được trang.
- LibSeat dành cho camera cố định. Camera cầm tay đi qua giảng đường đông cho kết quả yếu (xem `BAO_CAO.md`, mục 6.2).
