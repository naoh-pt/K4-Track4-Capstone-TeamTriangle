## Kết quả trên `seatanalysis_yolo11n_1280_c10` (2026-10-16 15:46)

- Dữ liệu: `data/seatanalysis`, 114 ảnh, 791 ghế có nhãn (37% có người). CALIB 383 ghế / TEST 408 ghế.
- Model: `yolo11n.pt`, conf=0.1, imgsz=1280, suy luận TB 199 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 74.3% | 67.4% | 13 | 92 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 75.7% | 69.6% | 15 | 84 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 76.2% | 70.0% | 13 | 84 | ov>0.2, chân<3.0 |

**Số đo chính:** 76.2% độ chính xác từng ghế trên 408 ghế (TEST, `seatanalysis_yolo11n_1280_c10`). Ma trận: TP=69, TN=242, FP=13, FN=84.
