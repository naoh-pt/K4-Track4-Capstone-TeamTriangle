## Kết quả trên `seatanalysis_yolo11n_640_c10` (2026-10-16 15:46)

- Dữ liệu: `data/seatanalysis`, 114 ảnh, 791 ghế có nhãn (37% có người). CALIB 383 ghế / TEST 408 ghế.
- Model: `yolo11n.pt`, conf=0.1, imgsz=640, suy luận TB 81 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 73.8% | 66.5% | 11 | 96 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 73.8% | 67.1% | 16 | 91 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 75.7% | 68.2% | 4 | 95 | ov>0.25, chân<0.5 |

**Số đo chính:** 75.7% độ chính xác từng ghế trên 408 ghế (TEST, `seatanalysis_yolo11n_640_c10`). Ma trận: TP=58, TN=251, FP=4, FN=95.
