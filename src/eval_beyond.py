"""
Script đánh giá độ chính xác (Vai C: Dữ liệu & Bằng chứng).
So sánh dự đoán của Pipeline với Nhãn Ground Truth trên dataset Beyond.
Xuất ma trận nhầm lẫn (Confusion Matrix), Accuracy từng ghế, và danh sách ca lỗi errors.csv.
"""

import glob
import os
import argparse
import pandas as pd
import numpy as np
from ultralytics import YOLO

def yolo_to_xyxy(line, w, h):
    parts = line.strip().split()
    cls = int(parts[0])
    bx, by, bw, bh = map(float, parts[1:])
    x1 = (bx - bw / 2.0) * w
    y1 = (by - bh / 2.0) * h
    x2 = (bx + bw / 2.0) * w
    y2 = (by + bh / 2.0) * h
    return cls, x1, y1, x2, y2

def overlap_ratio(seat, person):
    sx1, sy1, sx2, sy2 = seat
    px1, py1, px2, py2 = person
    iw = max(0.0, min(sx2, px2) - max(sx1, px1))
    ih = max(0.0, min(sy2, py2) - max(sy1, py1))
    inter = iw * ih
    seat_area = max(1e-6, (sx2 - sx1) * (sy2 - sy1))
    return inter / seat_area

def main():
    parser = argparse.ArgumentParser(description="Evaluate LibSeat on Beyond dataset")
    parser.add_argument("--img-dir", type=str, default="data/beyond/test/images")
    parser.add_argument("--lbl-dir", type=str, default="data/beyond/test/labels")
    parser.add_argument("--overlap-thr", type=float, default=0.25, help="Ngưỡng overlap diện tích ghế")
    parser.add_argument("--out-csv", type=str, default="results/errors.csv")
    args = parser.parse_args()

    print(">>> Đang tải YOLO model để đánh giá...")
    model = YOLO("yolo11n.pt")
    
    rows = []
    images = sorted(glob.glob(f"{args.img_dir}/*.jpg"))

    if not images:
        print(f"[Cảnh báo] Không tìm thấy ảnh trong {args.img_dir}")
        return

    print(f">>> Đang đánh giá trên {len(images)} ảnh kiểm thử...")

    for img_path in images:
        base_name = os.path.basename(img_path)
        lbl_path = os.path.join(args.lbl_dir, os.path.splitext(base_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue

        res = model(img_path, classes=[0], imgsz=480, device="cpu", verbose=False)[0]
        h, w = res.orig_shape
        persons = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []

        with open(lbl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                cls, sx1, sy1, sx2, sy2 = yolo_to_xyxy(line, w, h)
                seat_box = [sx1, sy1, sx2, sy2]
                
                # Dự đoán có người hay không
                pred_occ = any(overlap_ratio(seat_box, p) >= args.overlap_thr for p in persons)
                gt_occ = (cls == 1) # 1: Occupied, 0: Empty

                rows.append({
                    "image": base_name,
                    "seat_box": [round(x, 1) for x in seat_box],
                    "gt_occ": gt_occ,
                    "pred_occ": pred_occ,
                    "is_correct": (gt_occ == pred_occ)
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[Lỗi] Không có nhãn ghế nào được đọc.")
        return

    acc = df["is_correct"].mean()
    print("\n" + "="*50)
    print(f" KẾT QUẢ ĐÁNH GIÁ LIBSAT (BẰNG CHỨNG CHO VAI C)")
    print("="*50)
    print(f"Tổng số ghế được kiểm tra: {len(df)}")
    print(f"Độ chính xác từng ghế (Accuracy): {acc * 100:.2f}%")
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    ct = pd.crosstab(df["gt_occ"], df["pred_occ"], rownames=["Thực tế (GT)"], colnames=["Dự đoán (Pred)"])
    print(ct)
    print("="*50)

    # Xuất danh sách ca lỗi (Failure Cases)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    errors_df = df[df["is_correct"] == False]
    errors_df.to_csv(args.out_csv, index=False)
    print(f"Đã lưu {len(errors_df)} ca lỗi vào: {args.out_csv}")

if __name__ == "__main__":
    main()
