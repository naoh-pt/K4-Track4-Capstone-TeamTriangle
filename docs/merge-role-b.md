# Bàn giao nhánh B cho trưởng nhóm

Nhánh: `feat/role-b-geometry-smoothing`. Phần B cung cấp logic, chưa nối webcam/YOLO của A hoặc script đánh giá của C. Các đường dẫn A/C dưới đây là theo kế hoạch; cần đối chiếu tên file thực tế khi merge.

## Cách merge

Chạy khi working tree sạch, sau khi đã lấy nhánh A/C về nhánh tích hợp của nhóm:

```powershell
git fetch origin
git switch -c integration/abc origin/main
git merge origin/feat/role-b-geometry-smoothing
python -m pip install -r requirements-b.txt
python -m unittest discover -s tests -v
```

Nếu nhóm đã có nhánh tích hợp, chuyển sang nhánh đó thay cho lệnh tạo nhánh. Merge A/C vào cùng nhánh rồi sửa các điểm bên dưới. Sau khi test tích hợp và demo thành công, mở PR về `main` theo quy trình của nhóm.

## Các chỗ cần sửa để khớp A/C

| Chỗ cần đối chiếu | Thay đổi cần làm | Người phụ trách |
|---|---|---|
| `src/detect.py` của A | Đổi detection người thành `Person(tuple(float(v) for v in xyxy), confidence=float(conf))`; lọc đúng lớp person. Tọa độ là pixel `xyxy`, không phải `xywh` hoặc tọa độ chuẩn hóa | A |
| Khởi tạo trong `src/app.py` của A | Tạo danh sách `Seat(str(id), box)` đã xác nhận và một `OccupancyEngine` duy nhất trước vòng webcam | A+B |
| Vòng webcam trong `src/app.py` | Gọi `engine.update(people, time.monotonic())` sau inference. Dùng `[]` khi inference không thấy người, `None` khi inference lỗi/không chạy | A |
| Tracking ghế | Giữ ID ghế ổn định kể cả bị che; không tạo lại danh sách ghế từ detection mỗi khung. Khi ghế di chuyển, thay bbox với cùng ID trong `engine.seats`; chỉ giữ nguyên tập ID. Nếu thêm/bớt ghế, tạo engine mới và chấp nhận reset làm mượt | A+B |
| Resize/crop camera | Đồng bộ hệ tọa độ giữa bbox, anchor và ảnh hiệu chuẩn. Camera thay vị trí phải hiệu chuẩn lại | A+B |
| Hiển thị | Đỏ khi `occupied is True`, xanh khi `is False`, xám khi `is None`. Hiện `available/total` kèm `unknown`; không đổi unknown thành trống bằng `not occupied` | A |
| Log CSV | Một dòng/ghế/khung: thời gian, seat_id, raw_occupied, occupied, person_index, phiên bản và cấu hình ngưỡng. Quy ước `unknown` cho None; C cần parse rõ, không gọi `bool("False")` | A+C |
| `src/eval_beyond.py` của C | Dùng bbox nhãn làm `Seat`; gọi `assign_people` cho đánh giá ảnh độc lập. Không dùng trạng thái tích lũy qua các ảnh không liên tiếp | C |
| Đánh giá video | Tạo engine mới mỗi clip; dùng timestamp video `frame_index / fps` (hoặc timestamp thực của video), không dùng tốc độ xử lý offline làm thời gian bằng chứng | C |
| `requirements.txt` chung | Hợp nhất `numpy` và `opencv-python` từ `requirements-b.txt`, giữ dependencies YOLO/UI của A và phân tích của C. Cần Python 3.10+ | Trưởng nhóm |

## Mẫu nối vào vòng lặp của A

```python
import time
from src.geometry import Person, Seat, FloorCalibration
from src.occupancy import OccupancyEngine

# Một lần, từ danh sách ghế do A xác nhận. ID là ID ghế ổn định.
seats = [Seat("S1", (100, 150, 200, 300))]
engine = OccupancyEngine(seats, calibration=None)
# Khi có hiệu chuẩn: calibration=FloorCalibration.load("calibration.json")

# Mỗi lần inference chạy thành công, với result là Results của Ultralytics:
people = []
for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(),
                          result.boxes.cls.cpu().tolist(),
                          result.boxes.conf.cpu().tolist()):
    if int(cls) == 0:  # Xác nhận result.names[0] là person trên model của A.
        people.append(Person(tuple(box), float(conf)))
status = engine.update(people, time.monotonic())
for seat_id, row in status["seats"].items():
    state = row["occupied"]
    color = "gray" if state is None else ("red" if state else "green")
    # A dùng seat_id để tìm bbox và vẽ màu tương ứng.
```

Chạy app dạng module từ gốc repo, ví dụ `python -m src.app`, để relative import trong phần B hoạt động. Nếu A dùng Streamlit, giữ lệnh launcher của A và bảo đảm thư mục gốc repo nằm trên import path.

## Khi có xung đột file

- `.gitignore`: gộp quy tắc hai nhánh; giữ bỏ qua cache, môi trường ảo và file kế hoạch. Không stage dataset, model, log hoặc video chỉ vì merge.
- `src/__init__.py`: giữ các export/khởi tạo cần thiết của A/C; file của B chỉ chứa docstring.
- `src/geometry.py`, `src/smoothing.py`: nếu nhánh khác đã có logic cùng tên, hợp nhất theo API `Seat`, `Person`, `FloorCalibration`, `assign_people`, `SeatState`; sửa caller tương ứng. Không giữ hai bước làm mượt chạy nối tiếp vì sẽ tăng độ trễ.
- Đường dẫn package khác `src`: cập nhật import trong caller, `src/occupancy.py`, `src/calibrate.py` và tests nếu di chuyển module.
- Requirements: giữ phiên bản chung tương thích; không ghi đè toàn bộ file của A/C bằng file dependencies riêng của B.

## Điều kiện trước khi merge về main

1. Sáu unit test B đạt. Đây là kiểm tra logic, chưa chứng minh độ chính xác trên webcam.
2. Webcam đi hết detection → engine → màu ghế và số chỗ. Sau khởi động, `available + occupied + unknown == total`.
3. Người ngồi/đứng lên đổi trạng thái theo thời gian cấu hình; che ngắn hạn không làm nhấp nháy. Một người không chiếm hai ghế.
4. Detector lỗi không tự báo trống; giao diện báo dữ liệu cũ khi mất tín hiệu lâu. B giữ trạng thái cuối, A chịu trách nhiệm cảnh báo tuổi dữ liệu.
5. Thử đi ngang/đứng sau ghế và ghế bị kéo đi; ghi lại ca sai. Homography dùng điểm chạm sàn ước lượng, chưa phân biệt chắc chắn ngồi với đứng sát ghế.
6. C ghi rõ cấu hình, tập đánh giá và số đo thật. Không áp một homography chung cho ảnh nhiều camera; chỉ so làm mượt trên video có nhãn theo thời gian.

Các ngưỡng mặc định: overlap 0.3, confidence 0.25, khoảng cách 0.6 m, bật 0.5 s, tắt 1.0 s, khoảng ngắt bằng chứng tối đa 0.5 s. Nếu inference thường cách nhau hơn 0.5 s, tăng `max_gap_seconds` có chủ đích và đo lại. Hướng dẫn hiệu chuẩn chi tiết: [role-b.md](role-b.md).
