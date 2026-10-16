## Kết quả trên `beyond_yolo11n_960_c10` (2026-10-16 15:39)

- Dữ liệu: `data/beyond`, 233 ảnh, 3151 ghế có nhãn (12% có người). CALIB 1418 ghế / TEST 1733 ghế.
- Model: `yolo11n.pt`, conf=0.1, imgsz=960, suy luận TB 167 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 88.2% | 67.7% | 80 | 124 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 87.5% | 67.7% | 94 | 122 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 89.6% | 68.5% | 56 | 124 | ov>0.25, chân<2.5 |

**Số đo chính:** 89.6% độ chính xác từng ghế trên 1733 ghế (TEST, `beyond_yolo11n_960_c10`). Ma trận: TP=85, TN=1468, FP=56, FN=124.
