## So cấu hình bộ phát hiện trên `seatanalysis` (chọn theo CALIB)

| Model | imgsz | conf | ms/ảnh | Luật | CALIB bal. | TEST acc. | TEST bal. | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| yolo11n.pt | 640 | 0.25 | 78 | foot | 60.0% | 70.6% | 61.3% | 4 | 116 |
| yolo11n.pt | 640 | 0.1 | 81 | foot | 68.7% | 75.7% | 68.2% | 4 | 95 |
| yolo11n.pt | 960 | 0.1 | 127 | foot | 76.5% | 79.2% | 74.2% | 15 | 70 |
| yolo11n.pt | 1280 | 0.1 | 199 | foot | 74.0% | 76.2% | 70.0% | 13 | 84 |
| yolo11s.pt **(chọn)** | 960 | 0.1 | 272 | foot | 84.8% | 84.6% | 81.2% | 14 | 49 |
| yolo11s.pt | 1280 | 0.1 | 464 | foot | 76.0% | 77.5% | 71.9% | 15 | 77 |
