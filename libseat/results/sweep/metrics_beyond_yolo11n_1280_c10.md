## Kết quả trên `beyond_yolo11n_1280_c10` (2026-10-16 15:41)

- Dữ liệu: `data/beyond`, 233 ảnh, 3151 ghế có nhãn (12% có người). CALIB 1418 ghế / TEST 1733 ghế.
- Model: `yolo11n.pt`, conf=0.1, imgsz=1280, suy luận TB 301 ms/ảnh trên máy đo.
- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.

| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |
|---|---|---|---|---|---|
| Chỉ YOLO + overlap, ngưỡng mặc định 0.30 | 88.6% | 68.7% | 78 | 120 | ov>0.3 |
| Chỉ YOLO + overlap, ngưỡng đã chỉnh | 85.5% | 69.3% | 142 | 109 | ov>0.05 |
| + Luật điểm chân (gần đúng hình học sàn) | 87.4% | 69.9% | 107 | 111 | ov>0.05, chân<2.75 |

**Số đo chính:** 87.4% độ chính xác từng ghế trên 1733 ghế (TEST, `beyond_yolo11n_1280_c10`). Ma trận: TP=98, TN=1417, FP=107, FN=111.
