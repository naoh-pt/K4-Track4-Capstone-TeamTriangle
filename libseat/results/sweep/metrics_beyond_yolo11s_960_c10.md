## Kết quả trên `beyond_yolo11s_960_c10` (2026-10-16 15:42)

- Dữ liệu: `data/beyond`, 233 ảnh, 3151 ghế có nhãn (12% có người). CALIB 1418 ghế / TEST 1733 ghế.
- Model: `yolo11s.pt`, conf=0.1, imgsz=960, suy luận TB 376 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 88.2% | 68.7% | 86 | 119 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 84.9% | 68.7% | 151 | 110 | ov>0.05 |
| + Luật điểm chân (gần đúng hình học sàn) | 87.5% | 69.8% | 105 | 112 | ov>0.05, chân<2.75 |

**Số đo chính:** 87.5% độ chính xác từng ghế trên 1733 ghế (TEST, `beyond_yolo11s_960_c10`). Ma trận: TP=97, TN=1419, FP=105, FN=112.
