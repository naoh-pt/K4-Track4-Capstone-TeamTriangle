# Kế hoạch Vin Perception Hackathon: Đếm chỗ trống thư viện VinUni

**Track 4 · Capstone · Team 3 người**

> Mục tiêu: một demo computer vision chạy thật trên webcam, cho biết **ghế nào trong thư viện còn trống**, kèm một số đo, một ca lỗi, và dùng ít nhất 2 trụ cột kiến thức của Track 4.

---

## 1. Tóm tắt dự án

| Mục | Nội dung |
|---|---|
| **Tên** | LibSeat: bản đồ chỗ trống thư viện thời gian thực |
| **Đơn vị Vin** | VinUni (thư viện, phòng học) |
| **Người dùng** | Sinh viên tìm chỗ học; nhân viên thư viện theo dõi mức sử dụng |
| **Nỗi đau** | Đi vòng nhiều tầng, nhiều phòng để tìm chỗ trống, nhất là mùa thi |
| **Đầu vào** | Webcam laptop (demo trực tiếp) |
| **Model** | YOLO pretrained (COCO) cho `person` và `chair`, không train từ đầu |
| **Đầu ra** | Hình ảnh phòng có ghế tô xanh (trống) / đỏ (có người) và dòng "Còn X/Y chỗ" |
| **Trụ cột chính** | Hình học, Bất định (thêm Thời gian và Bằng chứng) |

---

## 2. Đối chiếu với đề (index.html)

### 2.1. Bốn yêu cầu bắt buộc

| Yêu cầu của đề | Cách đáp ứng |
|---|---|
| **Chạy thật** | Demo trực tiếp bằng webcam. Có video quay sẵn để dự phòng |
| **Dùng kiến thức của khóa (≥ 2 trụ cột)** | Hình học (homography), Bất định (làm mượt, ngưỡng, hysteresis), Thời gian (đo độ trễ) |
| **Một số đo, một ca lỗi** | Độ chính xác từng ghế trên bộ ảnh có nhãn. Ca lỗi: ghế bị che hoặc balo bị nhầm |
| **Dữ liệu an toàn** | Thành viên tự đóng vai. Ảnh công khai CC BY 4.0. Không dùng dữ liệu sinh viên thật |

### 2.2. Sáu tiêu chí chấm điểm

| Tiêu chí | Điểm | Chiến lược |
|---|---|---|
| Vận dụng kiến thức | 25 | Homography + làm mượt, giải thích vì sao chọn. **Đây là chỗ dễ mất điểm nhất** |
| Demo chạy được | 25 | Webcam trực tiếp, mời giám khảo tự ngồi hoặc xê dịch ghế |
| Độ hữu ích | 15 | Người dùng cụ thể (sinh viên VinUni, nhân viên thư viện) |
| Impact | 15 | Phép tính thời gian tiết kiệm, đường mở rộng, rủi ro |
| Bằng chứng | 10 | 1 số đo, 1 ca lỗi có nguyên nhân, nói rõ giới hạn |
| Trình bày và hỏi đáp | 10 | 10 phút đúng giờ, ai cũng trả lời được |

**Luật hòa điểm:** Demo chạy được cao hơn thắng trước, rồi đến Bằng chứng. Vì vậy phải ưu tiên demo ổn định.

---

## 3. Phân vai

| Vai | Người | Phụ trách | Trụ cột |
|---|---|---|---|
| **A: Pipeline và model** | ______ | Webcam, YOLO, tracking, giao diện hiển thị, giữ demo chạy ổn | Học, Thời gian |
| **B: Hình học và làm mượt** | ______ | Homography, gán người vào ghế, làm mượt trạng thái, ngưỡng | Hình học, Bất định |
| **C: Dữ liệu, đo, trình bày** | ______ | Tải dataset, đánh giá, ca lỗi, video dự phòng, slide, ước lượng tác động | Dữ liệu, Bằng chứng |

**Quy tắc chung:**
- Mỗi người phải hiểu cả hệ thống. Ở 1:45 mỗi người giải thích phần mình 2 phút cho hai bạn còn lại.
- Ai cũng có thể bị hội đồng hỏi, nên cuối buổi tự hỏi nhau 5 câu ở mục 11.
- Dùng chung một repo Git. Nhánh `main` luôn phải chạy được.

---

## 4. Kiến trúc hệ thống

```
Webcam ──► [A] YOLO (person, chair) ──► [A] Tracker (ID ổn định)
                                              │
                                              ▼
                       [B] Homography: ảnh ──► tọa độ mặt sàn
                                              │
                                              ▼
                       [B] Gán người ──► ghế (IoU + khoảng cách sàn)
                                              │
                                              ▼
                       [B] Làm mượt từng ghế (hysteresis theo thời gian)
                                              │
                                              ▼
                       [A] Hiển thị: ghế xanh/đỏ + "Còn X/Y chỗ" + log CSV
                                              │
                                              ▼
                       [C] So log với nhãn ──► độ chính xác, ca lỗi
```

### 4.1. Vị trí các trụ cột (phần này nói với giám khảo)

| Trụ cột | Nằm ở đâu | Vì sao chọn cách này |
|---|---|---|
| **Học** | Bước YOLO | Dùng model pretrained vì 2 giờ không đủ để train. COCO có sẵn `person` và `chair`. Nêu rõ model nhận ra gì và học từ dữ liệu nào |
| **Hình học** | Homography ảnh → mặt sàn | Camera nhìn xiên, nên người đứng phía sau có thể che hoặc trùng ghế phía trước trong ảnh. Trên mặt sàn thì khoảng cách có ý nghĩa thật |
| **Bất định** | Hysteresis và ngưỡng độ tin cậy | YOLO nhấp nháy giữa các khung. Nếu đổi trạng thái ngay thì số "còn X chỗ" nhảy liên tục |
| **Thời gian** | Đo độ trễ ngồi xuống → màn hình đổi màu | Có đánh đổi giữa độ nhạy và độ ổn định, phải nêu được con số |
| **Dữ liệu / Bằng chứng** | Bộ ảnh Beyond có nhãn, log CSV | Số đo kiểm chứng được và có ca lỗi |

### 4.2. Vì sao không dùng phương án khác (chuẩn bị cho câu hỏi hội đồng)

| Phương án | Lý do không chọn |
|---|---|
| Chỉ đếm tổng số người, không xét từng ghế | Không cho biết **ghế nào** trống, ít hữu ích |
| Vẽ vùng ghế trực tiếp trên ảnh, không homography | Sai khi góc camera xiên, người phía sau bị tính vào ghế phía trước |
| Train model riêng | 2 giờ quá ngắn, dữ liệu ít, dễ overfit |
| Dùng VLM (model ảnh + chữ) hỏi "ghế này có người không" | Chậm, khó đo chính xác, khó đạt thời gian thực |
| YOLO cỡ lớn (`l`, `x`) | Độ trễ cao trên webcam laptop, không cần thiết cho bài toán này |

---

## 5. Model và dữ liệu

### 5.1. Model

- **YOLO11n** (`yolo11n.pt`) từ Ultralytics, pretrained trên COCO (80 lớp, gồm `person` = 0 và `chair` = 56).
- Nếu laptop khỏe hoặc cần chính xác hơn: `yolo11s.pt`.
- Tracking: `model.track(..., tracker="bytetrack.yaml")` có sẵn trong Ultralytics.
- Nâng cấp tùy chọn nếu còn giờ: `yolo11n-pose.pt` để phân biệt ngồi và đứng.
- Lưu ý giấy phép: Ultralytics dùng AGPL-3.0, ổn cho hackathon, cần nhắc nếu triển khai thật.

### 5.2. Dataset công khai (để đo, không dùng để train)

| Dataset | Nội dung | Vai trò |
|---|---|---|
| **Beyond** (`lyeonsl/beyond-sub9e`, Roboflow Universe) | 120 ảnh thư viện, lớp `Empty Seat` và `Occupied Seat`, CC BY 4.0 | **Bộ đo chính** |
| **12dd02mm2025yyyy** (`seatanalysis`) | 114 ảnh, nhãn empty-chair, occupied-chair, standing-person | Bộ đo phụ (kiểm tra phòng khác) |
| **emptyseats** (`cybathlon`) | 948 ảnh, lớp chair, bag, person | Phân tích ca lỗi "balo trên ghế" |

**Cách tải:** đăng nhập Roboflow (miễn phí), mở trang dataset, bấm **Download Dataset**, chọn định dạng **YOLOv8**. Giải nén vào `data/beyond/`.

**Lưu ý bắt buộc:**
- Kiểm tra lại trang riêng của từng dataset về giấy phép trước khi đưa vào bài nộp.
- Xem nhanh 10-15 ảnh đầu của Beyond để chắc nhãn không sai nhiều.
- Nhãn là hộp bao quanh ghế (Empty Seat / Occupied Seat), không phải ghế do YOLO phát hiện. Cách đo ở mục 7 đã xử lý khác biệt này.

### 5.3. Dữ liệu cho demo trực tiếp

- 2-3 thành viên ngồi ở một bàn có 4-6 ghế trước webcam. Không cần quay sinh viên thật.
- Chuẩn bị 4 mốc trên sàn hoặc bàn (băng keo màu hoặc 4 marker ArUco in giấy) cho homography.
- Quay **video dự phòng** từ chính các thành viên đóng vai, chỉ lưu nội bộ.

---

## 6. Chuẩn bị trước buổi build (làm tối nay)

### Cả team
- [ ] Chốt đề tài, tên và vai A/B/C
- [ ] Tạo repo Git chung, cấu trúc thư mục như mục 12
- [ ] Mỗi máy cài: `pip install ultralytics opencv-python numpy pandas streamlit`
- [ ] Tải sẵn `yolo11n.pt` (chạy thử một lần để nó tự tải)

### Vai A
- [ ] Chạy thử YOLO trên webcam, ghi lại FPS
- [ ] Xác định máy nào sẽ dùng để demo và thử đúng máy đó

### Vai B
- [ ] Thử `cv2.getPerspectiveTransform` với 4 điểm click chuột trên một ảnh bất kỳ
- [ ] In 4 marker hoặc chuẩn bị băng keo màu

### Vai C
- [ ] Tạo tài khoản Roboflow, tải dataset Beyond và 12dd02mm2025yyyy
- [ ] Xem 10-15 ảnh đầu, ghi chú nếu nhãn lạ
- [ ] Dựng khung slide 6-8 trang

---

## 7. Lịch build chi tiết (mốc tính từ lúc phát đề)

### 0:00 - 0:15: Phát đề, chốt hướng
- Cả team xác nhận bài toán, người dùng, ai làm gì.
- Chốt **số đo**: độ chính xác từng ghế (đúng / sai) trên bộ ảnh có nhãn, và độ trễ cảnh báo.
- Chốt **ca lỗi dự kiến**: ghế bị che, balo trên ghế.

### 0:15 - 0:45: Vòng end-to-end thô (MỤC TIÊU QUAN TRỌNG NHẤT)

| A | B | C |
|---|---|---|
| Webcam + YOLO, vẽ hộp quanh người và ghế | Xác định ghế bằng hộp `chair` của YOLO, gán người vào ghế bằng IoU thô | Chạy script đánh giá (mục 8.3) trên Beyond, ra số liệu ban đầu |

**Cổng kiểm tra 0:45:** có thứ chạy từ webcam đến "Còn X/Y chỗ", dù xấu. Nếu chưa có, dừng mọi thứ khác để làm cho nó chạy.

### 0:45 - 1:15: Nối và ổn định

| A | B | C |
|---|---|---|
| Thêm tracking ByteTrack, khung hiển thị đẹp hơn, ghi log CSV | Làm mượt trạng thái từng ghế (mục 8.2) | Phân tích những ảnh sai, bắt đầu chọn ca lỗi |

### 1:15: **MỐC KIỂM TRA CỦA GIẢNG VIÊN**
Cần có: từ webcam đến "Còn X/Y chỗ", ổn định tạm được. Chuẩn bị 1 phút giải thích đang làm gì.

### 1:15 - 1:45: Thêm kiến thức khóa học vào hệ thống

| A | B | C |
|---|---|---|
| Đo độ trễ ngồi xuống → đổi màu; đo FPS | **Homography** ảnh → mặt sàn (mục 8.1); gán theo khoảng cách sàn | Chạy đánh giá lại với phiên bản mới, so **trước/sau** homography và làm mượt |

### 1:45 - 2:15: Củng cố, đo cuối, ghi chứng cứ

| A | B | C |
|---|---|---|
| Thử cho người lạ ngồi vào, kiểm tra ổn định; quay video dự phòng | Giải thích 2 trụ cột cho A và C (2 phút mỗi người) | Chốt số đo cuối, ca lỗi, ước lượng tác động, hoàn thiện slide |

### 2:15: **DỪNG CODE.** Không sửa thêm, chỉ chuẩn bị demo.

### 2:15 - 2:30: Chuẩn bị demo
- Chạy thử trên đúng máy, đúng phòng, đúng ánh sáng.
- Chốt video dự phòng. Tập dợt bài 10 phút một lần.
- Kiểm tra checklist ở mục 10.

### 2:30: Demo (tối đa 10 phút)

---

## 8. Hướng dẫn kỹ thuật

### 8.1. Homography (vai B)

Ý tưởng: click 4 điểm trên ảnh tương ứng 4 góc một hình chữ nhật đã biết kích thước trên sàn (ví dụ vùng 2 m × 1,5 m), rồi biến mọi điểm "chân người" và "chân ghế" sang tọa độ mét.

```python
import cv2
import numpy as np

# 4 điểm click trên ảnh (thứ tự: trái-trên, phải-trên, phải-dưới, trái-dưới)
img_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
# Tọa độ thật trên sàn (mét)
floor_pts = np.float32([[0, 0], [2.0, 0], [2.0, 1.5], [0, 1.5]])

H = cv2.getPerspectiveTransform(img_pts, floor_pts)

def to_floor(pt):
    p = np.array([[pt]], dtype=np.float32)
    return cv2.perspectiveTransform(p, H)[0, 0]  # (x_m, y_m)

def foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)  # giữa cạnh dưới của hộp
```

**Gán người vào ghế:** một ghế là "có người" nếu hộp người chồng hộp ghế (IoU hoặc tỉ lệ giao đủ lớn) **và** khoảng cách trên sàn giữa chân người và ghế nhỏ hơn ngưỡng (ví dụ 0,6 m). Điều kiện thứ hai là chỗ homography có tác dụng thật.

**Cách giải thích với giám khảo:** "Nếu chỉ dùng IoU trên ảnh, người đứng sau ghế sẽ trùng hộp với ghế. Chuyển sang tọa độ sàn thì ta thấy người đó cách ghế gần 1 m nên không tính."

### 8.2. Làm mượt bằng hysteresis (vai B)

```python
class SeatState:
    """Chỉ đổi trạng thái khi thấy đủ liên tục (giảm nhấp nháy)."""
    def __init__(self, on_frames=15, off_frames=30):
        self.occupied = False
        self.count = 0
        self.on_frames = on_frames    # cần thấy "có người" liên tục bấy nhiêu khung
        self.off_frames = off_frames  # cần thấy "trống" liên tục bấy nhiêu khung

    def update(self, detected: bool) -> bool:
        if detected != self.occupied:
            self.count += 1
            need = self.on_frames if detected else self.off_frames
            if self.count >= need:
                self.occupied = detected
                self.count = 0
        else:
            self.count = 0
        return self.occupied
```

- Đặt `off_frames` lớn hơn `on_frames`: tránh ghế bị coi là trống khi người cúi xuống hoặc bị che thoáng qua.
- Con số `on_frames`, `off_frames` là **tham số cần chỉnh** dựa trên FPS thực tế. Ghi lại giá trị đã chọn và lý do.
- Có thể thay bằng trung bình trượt (EMA) của điểm tin cậy nếu muốn mượt hơn.

### 8.3. Script đánh giá trên dataset có nhãn (vai C)

Cách đo không phụ thuộc vào việc YOLO có phát hiện đúng ghế hay không: **dùng hộp ghế của nhãn làm vùng ghế**, rồi xem YOLO phát hiện người có chồng vào vùng đó không.

```python
# eval_beyond.py (khung, cần chỉnh theo thư mục và data.yaml thật)
import glob, os
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
IMG_DIR = "data/beyond/test/images"
LBL_DIR = "data/beyond/test/labels"
OCC_ID, EMPTY_ID = 1, 0   # ĐỌC data.yaml để xác nhận lại id của hai lớp!
OVERLAP_THR = 0.3         # tỉ lệ diện tích ghế bị người che, tham số cần chỉnh

def yolo_to_xyxy(line, w, h):
    c, x, y, bw, bh = map(float, line.split())
    return int(c), (x - bw/2) * w, (y - bh/2) * h, (x + bw/2) * w, (y + bh/2) * h

def overlap_ratio(seat, person):
    sx1, sy1, sx2, sy2 = seat
    px1, py1, px2, py2 = person
    iw = max(0, min(sx2, px2) - max(sx1, px1))
    ih = max(0, min(sy2, py2) - max(sy1, py1))
    return (iw * ih) / max(1e-6, (sx2 - sx1) * (sy2 - sy1))

rows = []
for img in sorted(glob.glob(f"{IMG_DIR}/*.jpg")):
    res = model(img, classes=[0], verbose=False)[0]      # chỉ lớp person
    h, w = res.orig_shape
    persons = res.boxes.xyxy.cpu().numpy().tolist()
    lbl = os.path.join(LBL_DIR, os.path.splitext(os.path.basename(img))[0] + ".txt")
    if not os.path.exists(lbl):
        continue
    for line in open(lbl):
        cls, *box = yolo_to_xyxy(line, w, h)
        pred_occ = any(overlap_ratio(box, p) > OVERLAP_THR for p in persons)
        rows.append((os.path.basename(img), cls == OCC_ID, pred_occ))

import pandas as pd
df = pd.DataFrame(rows, columns=["image", "gt_occ", "pred_occ"])
acc = (df.gt_occ == df.pred_occ).mean()
print(f"Số ghế: {len(df)} | Độ chính xác từng ghế: {acc:.3f}")
print(pd.crosstab(df.gt_occ, df.pred_occ, rownames=["Nhãn"], colnames=["Dự đoán"]))
df[df.gt_occ != df.pred_occ].to_csv("errors.csv", index=False)  # danh sách ca lỗi
```

**Cách dùng kết quả:**
- Báo cáo **độ chính xác từng ghế**, ma trận nhầm lẫn (bao nhiêu ghế bị bỏ sót, bao nhiêu bị đếm nhầm).
- Mở `errors.csv`, xem 5-10 ảnh sai, nhóm theo nguyên nhân (bị che, góc nhìn, balo, ánh sáng). Chọn **một** ca tiêu biểu để trình bày.
- Chỉnh `OVERLAP_THR` trên một nửa ảnh, **đo cuối trên nửa còn lại** để con số đáng tin.
- Chạy lại sau khi thêm mỗi cải tiến để có bảng "trước/sau".

### 8.4. Đo độ trễ (vai A)

- Ghi `time.time()` mỗi khung vào log CSV cùng trạng thái ghế thô và trạng thái đã làm mượt.
- Dùng đồng hồ bấm giờ: người ngồi xuống, bấm giờ cho đến khi màn hình đổi màu. Lặp lại 5-10 lần, lấy trung bình và độ lệch.
- Báo cáo: FPS trung bình, độ trễ ngồi xuống → đổi màu, và đánh đổi do `on_frames` / `off_frames`.

---

## 9. Bằng chứng (để ăn trọn 10 điểm)

| Thành phần | Nội dung cần có |
|---|---|
| **Một số đo** | Độ chính xác từng ghế trên N ghế (ghi rõ N, bộ dữ liệu, phiên bản hệ thống) |
| **Một ca lỗi** | Một ảnh cụ thể hệ thống sai, kèm nguyên nhân (ví dụ ghế bị người khác che) |
| **Giới hạn** | Ảnh công khai là thư viện khác, chưa chứng minh được trên phòng thật của VinUni; không phân biệt "giữ chỗ bằng đồ" với trống; chưa thử nhiều điều kiện ánh sáng |
| **Trước/sau** | Bảng so sánh: chỉ YOLO + IoU vs thêm homography vs thêm làm mượt |

Mẫu bảng "trước/sau" cần điền số thật khi chạy:

| Phiên bản | Độ chính xác từng ghế | Ghi chú |
|---|---|---|
| Chỉ YOLO + IoU thô | ____ % | Điểm xuất phát |
| + Homography (khoảng cách sàn) | ____ % | Giảm lỗi người đứng sau ghế |
| + Làm mượt hysteresis | ____ % (trên video) | Số nhấp nháy giảm từ ____ xuống ____ |

---

## 10. Ước lượng tác động (Impact, 15 điểm)

**Công thức mẫu (các số bên dưới là giả định, phải thay bằng số khảo sát nhanh trong lớp trước khi nói):**

```
Thời gian tiết kiệm / ngày =
   (số sinh viên dùng thư viện / ngày) × (phút tìm chỗ trung bình giảm đi)

Ví dụ giả định: 500 sinh viên × 5 phút = 2.500 phút ≈ 42 giờ / ngày
```

**Việc cần làm:** khảo sát nhanh 10-15 bạn: "Bạn mất bao nhiêu phút tìm chỗ trống?" và "Mỗi ngày bạn vào thư viện mấy lần?". Dùng số thật đó, ghi rõ cỡ mẫu nhỏ.

**Đường mở rộng:**
1. Một phòng → nhiều camera trong cả thư viện, ghép thành bản đồ chung
2. Hiển thị trên web/app cho sinh viên, màn hình ở cửa thư viện
3. Thống kê giờ cao điểm cho nhân viên thư viện

**Rủi ro khi đưa vào thực tế (phải nêu):**
- **Quyền riêng tư:** chỉ đếm và định vị ghế, không lưu hình ảnh, không nhận diện danh tính, làm mờ hoặc xử lý ngay tại thiết bị
- Cần sự đồng ý và thông báo cho người dùng thư viện
- Độ chính xác giảm khi camera bị che hoặc thiếu sáng
- Chi phí và bảo trì camera ở nhiều tầng

---

## 11. Bài thuyết trình 10 phút

| Thời lượng | Nội dung | Người nói |
|---|---|---|
| **1 phút** | Bài toán: sinh viên VinUni mất thời gian tìm chỗ. Người dùng và giá trị | C |
| **4 phút** | **Demo trực tiếp.** Mời giám khảo ngồi vào hoặc xê dịch ghế, màn hình đổi màu | A điều khiển, B giải thích khi cần |
| **2 phút** | Số đo, ca lỗi, 2 trụ cột và vị trí trong hệ thống, tác động ước lượng | C (số đo), B (trụ cột) |
| **3 phút** | Hỏi đáp | Cả nhóm |

### Kịch bản demo 4 phút
1. (30 giây) Phòng trống, hiện "Còn X/X chỗ".
2. (60 giây) Một thành viên ngồi vào ghế, ghế đổi sang đỏ, số chỗ giảm.
3. (60 giây) Mời giám khảo tự ngồi hoặc kéo ghế sang vị trí khác.
4. (60 giây) Người đi ngang qua nhưng không ngồi: hệ thống không đổi (minh họa Bất định và Hình học).
5. (30 giây) Một ca hệ thống sai có chủ đích (ví dụ che ghế bằng người khác) và giải thích lý do.

**Nếu demo hỏng:** chuyển sang video quay sẵn và **nói rõ đó là video**. Ca lỗi có chủ đích vẫn được tính điểm vì là bằng chứng.

### Câu hỏi hội đồng có thể hỏi (mọi người đều phải trả lời được)
1. Nếu chưa học Track 4 thì sản phẩm này làm được không? Phần nào cần kiến thức khóa?
2. Vì sao dùng homography mà không vẽ vùng trực tiếp trên ảnh?
3. Tại sao chọn `on_frames` và `off_frames` này? Đổi sẽ thế nào?
4. Độ chính xác đo như thế nào? Có thể tin nó đến đâu?
5. Hệ thống sai khi nào? Giải thích ca lỗi bạn đã trình bày.
6. Làm sao xử lý quyền riêng tư nếu triển khai thật?
7. Vì sao chọn YOLO nhỏ mà không chọn model lớn hơn?
8. Nếu mở rộng ra cả thư viện thì cần thay đổi gì?

---

## 12. Cấu trúc repo gợi ý

```
libseat/
├── README.md
├── requirements.txt
├── data/
│   ├── beyond/            # dataset Roboflow đã giải nén
│   └── logs/              # log CSV khi chạy demo
├── src/
│   ├── detect.py          # [A] YOLO + tracking
│   ├── geometry.py        # [B] homography, gán người vào ghế
│   ├── smoothing.py       # [B] SeatState
│   ├── app.py             # [A] webcam + hiển thị
│   └── eval_beyond.py     # [C] đánh giá
├── results/
│   ├── errors.csv
│   └── metrics.md         # bảng số đo cuối
└── demo_backup.mp4        # video dự phòng (không công khai)
```

---

## 13. Rủi ro và cách xử lý

| Rủi ro | Mức | Xử lý |
|---|---|---|
| Không có vòng end-to-end đến 0:45 | Cao | Ưu tiên tuyệt đối, bỏ homography tạm, chạy bản thô trước |
| Webcam hoặc ánh sáng phòng demo khác lúc thử | Trung bình | Thử đúng phòng ở 2:15, có video dự phòng |
| YOLO chạy chậm trên laptop | Trung bình | Dùng `yolo11n`, giảm độ phân giải ảnh, giảm tần suất suy luận |
| Nhãn dataset sai hoặc lệch định dạng | Trung bình | Xem nhanh 10-15 ảnh đầu, đọc `data.yaml` xác nhận id lớp |
| Bị chấm "chỉ gọi model có sẵn" | Cao | Làm cho được homography và làm mượt, trình bày lý do chọn |
| Quá giờ 10 phút | Trung bình | Tập dợt có bấm giờ, cắt bớt phần nói chứ không cắt demo |
| Một người không trả lời được câu hỏi | Trung bình | Cross-training ở 1:45, tự hỏi nhau 8 câu ở mục 11 |

---

## 14. Checklist trước khi lên demo (9 mục của đề)

- [ ] Demo chạy trên **đúng máy** sẽ dùng, đã thử webcam ở phòng demo
- [ ] Có **video quay sẵn** để dự phòng
- [ ] Có **một số đo** và cả team nói được nó đo thế nào
- [ ] Có **ít nhất một ca lỗi** kèm lý do
- [ ] Chỉ ra được **ít nhất 2 trụ cột** và vị trí của chúng trong hệ thống
- [ ] **Không dùng dữ liệu thật** của sinh viên hoặc bệnh nhân
- [ ] Mỗi thành viên trả lời được **vì sao team chọn cách làm này**
- [ ] Nói được **người dùng là ai** và họ được gì
- [ ] Có **ước lượng tác động** nếu triển khai ở Vin, dù chỉ là phép tính đơn giản

---

## 15. Phụ lục: tổng hợp các điểm cần tự kiểm tra

Các thông tin dưới đây được tổng hợp từ nguồn công khai và hiểu biết chung, **cần kiểm tra lại trước khi đưa vào bài nộp:**

- Giấy phép của từng dataset trên Roboflow (xem trang riêng của từng bộ)
- Id lớp trong `data.yaml` của Beyond (`Empty Seat` và `Occupied Seat` mang số nào)
- Chỉ số FPS và độ trễ thực tế trên máy demo (chỉ có thể biết khi chạy thử)
- Các con số trong mục tác động (đều là giả định đến khi có số khảo sát thật)
- Tên lớp COCO: `person` = 0, `chair` = 56 (xác nhận bằng `model.names`)
