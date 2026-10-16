## Kết quả trên `beyond_yolo11s_1280_c10` (2026-10-16 15:45)

- Dữ liệu: `data/beyond`, 233 ảnh, 3151 ghế có nhãn (12% có người). CALIB 1418 ghế / TEST 1733 ghế.
- Model: `yolo11s.pt`, conf=0.1, imgsz=1280, suy luận TB 698 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 88.4% | 68.6% | 81 | 120 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 87.6% | 68.6% | 97 | 118 | ov>0.2 |
| + Luật điểm chân (gần đúng hình học sàn) | 89.3% | 69.4% | 66 | 119 | ov>0.2, chân<2.75 |

**Số đo chính:** 89.3% độ chính xác từng ghế trên 1733 ghế (TEST, `beyond_yolo11s_1280_c10`). Ma trận: TP=90, TN=1458, FP=66, FN=119.
