# LibSeat: số đo cuối (vai C)

Cập nhật: 2026-10-16. Mọi số dưới đây đo trên **nửa TEST**; ngưỡng và cấu hình được chọn trên **nửa CALIB** (chia cố định theo hash tên file, xem `src/eval_beyond.py`).

## Dữ liệu

| Bộ | Nguồn | Ảnh | Ghế có nhãn | % có người | Giấy phép |
|---|---|---|---|---|---|
| seatanalysis v1 (camera giám sát văn phòng cố định) | https://universe.roboflow.com/seatanalysis/12dd02mm2025yyyy/dataset/1 | 114 | 791 | 37% | CC BY 4.0 |
| Beyond v3 (ảnh điện thoại, thư viện) | https://universe.roboflow.com/lyeonsl/beyond-sub9e/dataset/3 | 233 | 3.151 | 12% | CC BY 4.0 |

Cách đo: dùng hộp ghế của **nhãn** làm vùng ghế, xem YOLO có thấy người chồng lên không. Số đo không phụ thuộc YOLO có thấy ghế hay không.

## Số đo chính (cấu hình chọn: YOLO11s, imgsz 960, conf 0.10, luật chồng hộp + điểm chân)

| Bộ | Ghế TEST | Độ chính xác | **Balanced acc.** | Bắt được ghế có người (recall) | Báo "có người" đúng (precision) | TP / TN / FP / FN |
|---|---|---|---|---|---|---|
| seatanalysis | 408 | 84,6% | **81,2%** | 68,0% | 88,1% | 104 / 241 / 14 / 49 |
| Beyond | 1.733 | 87,5% | **69,8%** | 46,4% | 48,0% | 97 / 1419 / 105 / 112 |
| Beyond, bỏ 20 ảnh xoay 90° | 1.565 | 88,1% | 71,0% | 48,9% | 48,1% | 88 / 1290 / 95 / 92 |

**Câu nói khi trình bày:** "Trên camera cố định, LibSeat xác định đúng trạng thái 84,6% trong 408 ghế, balanced accuracy 81%. Trên ảnh điện thoại khó hơn, balanced accuracy 70%."

**Không báo riêng "89% chính xác" của Beyond:** 88% ghế ở đó trống, đoán "trống" cho tất cả cũng được 88%.

## Bảng trước/sau (balanced accuracy, TEST)

| Phiên bản | seatanalysis | Beyond |
|---|---|---|
| YOLO11n 640, conf 0.25, chồng hộp ngưỡng 0.30 (điểm xuất phát) | 62,2% | 66,0% |
| YOLO11n 640, + luật điểm chân | 61,3% | 66,8% |
| YOLO11s 960, conf 0.10, chồng hộp ngưỡng 0.30 | 80,5% | 68,7% |
| **YOLO11s 960, conf 0.10, + luật điểm chân (cuối)** | **81,2%** | **69,8%** |

So 6 cấu hình bộ phát hiện: `sweep_seatanalysis.md`, `sweep_beyond.md`.

**Kết luận trung thực:**
- Phần lớn cải thiện đến từ **cấu hình bộ phát hiện** (+19 điểm trên seatanalysis), không phải từ luật gán.
- **Luật điểm chân trên ảnh tĩnh chỉ thêm khoảng 0,4–1 điểm.** Đây là bản gần đúng của homography (ảnh không có 4 mốc sàn, Beyond lại bị kéo giãn 960×960). Lợi ích của homography thật phải được chứng minh trên demo webcam.

## Ca lỗi

Gần như toàn bộ FN (218/229 ở seatanalysis, 219/236 ở Beyond, cấu hình gốc) có overlap = 0: **YOLO không thấy người**. Nguyên nhân quan sát được:
1. Người ở xa, nhỏ (khoảng 20–30 px chiều cao).
2. Người ngồi sau màn hình, chỉ lộ đầu (camera nhìn từ trên cao).
3. 20/233 ảnh Beyond bị xoay 90° (kiểm tra bằng mắt, danh sách: `rotated_beyond_manual.csv`).

**Ca lỗi trình bày** (`failure_case_beyond.jpg`, ảnh `IMG_5600`): người đứng sát ghế trống. Hộp người che 57% ghế, chân cách đáy ghế 0,4 lần chiều cao ghế, nên cả hai luật báo "có người" (FP). Trong cùng ảnh, hai ghế phía sau bị che 63–70% nhưng chân ở xa, được luật điểm chân báo "trống" đúng. Hướng sửa: pose (ngồi/đứng), làm mượt theo thời gian.

Ảnh tổng hợp lỗi: `sweep/errors_*_yolo11s_960_c10/contact_sheet.jpg`.

## Giới hạn

- seatanalysis là các khung của **cùng một camera**, nên CALIB và TEST cùng cảnh: số lạc quan.
- Cả hai bộ là phòng khác, chưa chứng minh trên phòng thật của VinUni.
- Không phân biệt "giữ chỗ bằng đồ" với ghế trống.

## Lưu ý cho vai A (độ trễ)

Thời gian suy luận đo trên CPU laptop này (không có GPU dùng được):

| Cấu hình | ms/ảnh | ≈ FPS |
|---|---|---|
| YOLO11n 640 | ~90 | ~11 |
| YOLO11n 960 | ~150 | ~7 |
| YOLO11s 960 (chính xác nhất) | ~320 | ~3 |

Cấu hình chính xác nhất có thể quá chậm cho demo webcam. Gợi ý: demo dùng YOLO11n 640–960 với conf 0.10 (seatanalysis: 68–74% balanced acc.), người ngồi gần camera nên ít bị nhỏ. `on_frames`/`off_frames` của vai B phải tính theo FPS thật.

## Chưa có (cần dữ liệu từ demo / khảo sát)

- FPS, nhấp nháy trước/sau làm mượt, độ trễ ngồi xuống → đổi màu: `src/eval_log.py` (cần log của A).
- Impact: `src/impact.py` (cần khảo sát 10–15 bạn).

## Tái lập

```
python src/eval_beyond.py --data data/beyond --tag beyond
python src/sweep_detector.py --data data/beyond --tag beyond
python src/sweep_detector.py --data data/seatanalysis --tag seatanalysis
python src/show_errors.py --data data/beyond --tag beyond_yolo11s_960_c10 --model yolo11s.pt --conf 0.1 --imgsz 960 --out results/sweep
```
