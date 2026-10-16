# Phần B: hình học và trạng thái ghế

Trưởng nhóm xem [hướng dẫn merge và các điểm phải sửa với A/C](merge-role-b.md).

Cài tại thư mục gốc repo:

```powershell
python -m pip install -r requirements-b.txt
python -m unittest discover -s tests -v
```

## Tích hợp với A

```python
import time
from src.geometry import Seat, Person, FloorCalibration
from src.occupancy import OccupancyEngine

# Khởi tạo một lần từ các ghế đã xác nhận lúc thiết lập camera.
engine = OccupancyEngine(
    [Seat("S1", (100, 150, 200, 300))],
    # calibration=FloorCalibration.load("calibration.json"),
    overlap_threshold=0.3,
    max_distance_m=0.6,
    on_seconds=0.5,
    off_seconds=1.0,
)

# Trong mỗi khung hình: đổi tensor YOLO thành tuple số Python.
people = [Person((110, 100, 210, 310), confidence=0.9)]
result = engine.update(people, time.monotonic())
print(result)
```

Không khởi tạo lại engine mỗi khung. `person_index` là vị trí trong danh sách người của khung hiện tại, không phải tracking ID. Ghế có ID ổn định do A cấp. Dùng cùng hệ tọa độ pixel cho ghế, người và ảnh hiệu chuẩn; không resize ảnh sau hiệu chuẩn nếu chưa đổi tọa độ tương ứng.

`[]` nghĩa là đã suy luận nhưng không thấy người; `None` nghĩa là không có bằng chứng vì suy luận lỗi hoặc bỏ khung. Tín hiệu thiếu giữ trạng thái cuối và hủy đếm liên tục. Khoảng cách giữa hai lần cập nhật lớn hơn `max_gap_seconds` cũng hủy đếm. Chỉnh tham số này nếu inference chậm. Khi camera mất tín hiệu lâu, A cần hiển thị cảnh báo dữ liệu cũ; trạng thái giữ lại không chứng minh ghế hiện vẫn trống.

Đầu ra gồm `seats[id].raw_occupied`, `seats[id].occupied`, `person_index`, `available`, `occupied`, `unknown`, `total`. `occupied=None` khi chưa đủ bằng chứng ban đầu. Giao diện nên dùng màu xám và hiện số ghế chưa xác định. CSV cho C nên có timestamp, seat_id, raw_occupied, occupied và phiên bản cấu hình.

Danh sách ghế tồn tại xuyên suốt dù detector mất hộp ghế. Khi ghế được kéo đi, A phải cập nhật `engine.seats` với hộp mới và cùng ID, không thêm/bớt ID trực tiếp vì bộ trạng thái được tạo lúc khởi tạo. Nếu bố trí thay đổi hoàn toàn, tạo engine mới.

## Hiệu chuẩn

Đặt bốn mốc cùng trên mặt sàn tạo hình chữ nhật đã đo. Chụp ảnh từ đúng camera, đúng độ phân giải demo:

```powershell
python -m src.calibrate room.jpg --width 2 --height 1.5 --output calibration.json
```

Click góc trái trên, phải trên, phải dưới, trái dưới của hình chữ nhật. Enter lưu, R chọn lại, Esc hủy. Không lấy mốc trên bàn rồi chuyển chân người trên sàn. Khi camera di chuyển phải hiệu chuẩn lại. Kích thước là mét; file JSON lưu điểm ảnh và kích thước để tái tạo ma trận.

Mặc định dùng giữa cạnh dưới bounding box làm điểm chân. Có thể truyền `floor_anchor=(x,y)` bằng tọa độ ảnh cho từng Seat/Person nếu có điểm tiếp xúc sàn tốt hơn. Khi bị che, điểm này có thể sai. Homography chỉ đúng với điểm thuộc mặt phẳng đã hiệu chuẩn và không tự phân biệt người đứng sát ghế với người ngồi.

Ghép cặp theo tỉ lệ diện tích ghế bị người che, lọc khoảng cách sàn nếu có hiệu chuẩn, rồi chọn tham lam theo overlap giảm dần. Mỗi người và mỗi ghế chỉ tham gia một cặp. Đây là heuristic, có thể sai khi nhiều người/ghế chồng nhau; không phải tối ưu ghép cặp toàn cục.

## Bàn giao và đo

Các ngưỡng hiện là giá trị khởi đầu, chưa được chứng minh trên webcam. Cùng A thử phòng trống, ngồi xuống/đứng lên, đi ngang, đứng sau ghế và che ngắn hạn. Ghi độ trễ đổi màu và số lần trạng thái nhấp nháy, so với bản raw. Với C: dùng ảnh để đo overlap; đo làm mượt trên chuỗi video có nhãn thời gian. Chỉ đánh giá homography trên camera có hiệu chuẩn tương ứng, không áp một ma trận lên mọi ảnh dataset.
