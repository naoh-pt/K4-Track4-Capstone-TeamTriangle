## Kết quả trên `seatanalysis_yolo11s_1280_c10` (2026-10-16 15:48)

- Dữ liệu: `data/seatanalysis`, 114 ảnh, 791 ghế có nhãn (37% có người). CALIB 383 ghế / TEST 408 ghế.
- Model: `yolo11s.pt`, conf=0.1, imgsz=1280, suy luận TB 464 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 77.5% | 71.5% | 12 | 80 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 77.0% | 71.5% | 17 | 77 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 77.5% | 71.9% | 15 | 77 | ov>0.2, chân<3.0 |

**Số đo chính:** 77.5% độ chính xác từng ghế trên 408 ghế (TEST, `seatanalysis_yolo11s_1280_c10`). Ma trận: TP=76, TN=240, FP=15, FN=77.
