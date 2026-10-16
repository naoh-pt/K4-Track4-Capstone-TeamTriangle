## Kết quả trên `seatanalysis_yolo11n_960_c10` (2026-10-16 15:46)

- Dữ liệu: `data/seatanalysis`, 114 ảnh, 791 ghế có nhãn (37% có người). CALIB 383 ghế / TEST 408 ghế.
- Model: `yolo11n.pt`, conf=0.1, imgsz=960, suy luận TB 127 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 75.7% | 69.5% | 14 | 85 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 79.2% | 74.3% | 16 | 69 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 79.2% | 74.2% | 15 | 70 | ov>0.2, chân<3.0 |

**Số đo chính:** 79.2% độ chính xác từng ghế trên 408 ghế (TEST, `seatanalysis_yolo11n_960_c10`). Ma trận: TP=83, TN=240, FP=15, FN=70.
