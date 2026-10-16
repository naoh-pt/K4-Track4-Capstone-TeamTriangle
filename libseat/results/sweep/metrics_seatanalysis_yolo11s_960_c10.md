## Kết quả trên `seatanalysis_yolo11s_960_c10` (2026-10-16 15:47)

- Dữ liệu: `data/seatanalysis`, 114 ảnh, 791 ghế có nhãn (37% có người). CALIB 383 ghế / TEST 408 ghế.
- Model: `yolo11s.pt`, conf=0.1, imgsz=960, suy luận TB 272 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 84.3% | 80.5% | 11 | 53 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 84.1% | 80.8% | 16 | 49 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 84.6% | 81.2% | 14 | 49 | ov>0.2, chân<3.0 |

**Số đo chính:** 84.6% độ chính xác từng ghế trên 408 ghế (TEST, `seatanalysis_yolo11s_960_c10`). Ma trận: TP=104, TN=241, FP=14, FN=49.
