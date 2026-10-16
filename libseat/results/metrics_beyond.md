## Kết quả trên `beyond` (2026-10-16 15:36)

- Dữ liệu: `data/beyond`, 233 ảnh, 3151 ghế có nhãn (12% có người). CALIB 1418 ghế / TEST 1733 ghế.
- Model: `yolo11n.pt`, conf=0.25, imgsz=640, suy luận TB 117 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 88.1% | 66.0% | 75 | 132 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 87.5% | 66.3% | 88 | 129 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 89.2% | 66.8% | 56 | 131 | ov>0.2, chân<2.75 |

**Số đo chính:** 89.2% độ chính xác từng ghế trên 1733 ghế (TEST, `beyond`). Ma trận: TP=78, TN=1468, FP=56, FN=131.
