## So cấu hình bộ phát hiện trên `beyond` (chọn theo CALIB)

| Model | imgsz | conf | ms/ảnh | Luật | CALIB bal. | TEST acc. | TEST bal. | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| yolo11n.pt | 640 | 0.25 | 103 | foot | 67.8% | 89.2% | 66.8% | 56 | 131 |
| yolo11n.pt | 640 | 0.1 | 97 | foot | 68.3% | 88.7% | 68.6% | 75 | 121 |
| yolo11n.pt | 960 | 0.1 | 167 | foot | 68.0% | 89.6% | 68.5% | 56 | 124 |
| yolo11n.pt | 1280 | 0.1 | 301 | foot | 68.6% | 87.4% | 69.9% | 107 | 111 |
| yolo11s.pt **(chọn)** | 960 | 0.1 | 376 | foot | 69.9% | 87.5% | 69.8% | 105 | 112 |
| yolo11s.pt | 1280 | 0.1 | 698 | foot | 69.5% | 89.3% | 69.4% | 66 | 119 |
