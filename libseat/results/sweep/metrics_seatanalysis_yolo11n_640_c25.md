## Kết quả trên `seatanalysis_yolo11n_640_c25` (2026-10-16 15:45)

- Dữ liệu: `data/seatanalysis`, 114 ảnh, 791 ghế có nhãn (37% có người). CALIB 383 ghế / TEST 408 ghế.
- Model: `yolo11n.pt`, conf=0.25, imgsz=640, suy luận TB 78 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 70.6% | 62.2% | 11 | 109 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 69.9% | 61.0% | 9 | 114 | ov>0.4 |
| + Luật điểm chân (gần đúng hình học sàn) | 70.6% | 61.3% | 4 | 116 | ov>0.4, chân<0.5 |

**Số đo chính:** 70.6% độ chính xác từng ghế trên 408 ghế (TEST, `seatanalysis_yolo11n_640_c25`). Ma trận: TP=37, TN=251, FP=4, FN=116.
