## Kết quả trên `beyond_yolo11n_640_c10` (2026-10-16 15:39)

- Dữ liệu: `data/beyond`, 233 ảnh, 3151 ghế có nhãn (12% có người). CALIB 1418 ghế / TEST 1733 ghế.
- Model: `yolo11n.pt`, conf=0.1, imgsz=640, suy luận TB 97 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 87.7% | 68.0% | 92 | 121 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 86.8% | 67.9% | 110 | 119 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 88.7% | 68.6% | 75 | 121 | ov>0.2, chân<2.75 |

**Số đo chính:** 88.7% độ chính xác từng ghế trên 1733 ghế (TEST, `beyond_yolo11n_640_c10`). Ma trận: TP=88, TN=1449, FP=75, FN=121.
