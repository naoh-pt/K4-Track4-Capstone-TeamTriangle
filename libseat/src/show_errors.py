"""[C] Vẽ các ca lỗi để chọn 1 ca tiêu biểu đưa lên slide.

Đọc results/errors_<tag>.csv (do eval_beyond.py sinh ra), vẽ lại ảnh:
  - hộp ghế nhãn: xanh = trống, đỏ = có người (theo NHÃN)
  - ghế bị đoán sai được viền vàng dày và ghi FP/FN
  - hộp người YOLO: trắng mảnh, chấm tròn = điểm chân
Ghi ra results/errors_<tag>/*.jpg và một ảnh lưới contact_sheet.jpg.

Chạy:
    python src/show_errors.py --data data/beyond --tag beyond --max 12
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/beyond")
    ap.add_argument("--tag", default="beyond")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    seats = pd.read_csv(os.path.join(args.out, f"seats_{args.tag}.csv"))
    err_imgs = (seats[seats.error.notna() & (seats.error != "")]
                .groupby("image").size().sort_values(ascending=False))
    out_dir = os.path.join(args.out, f"errors_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    tiles = []
    for name in err_imgs.index[: args.max]:
        path = os.path.join(args.data, name)
        img = cv2.imread(path)
        if img is None:
            continue
        res = model(path, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy().astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.circle(img, ((x1 + x2) // 2, y2), 4, (255, 255, 255), -1)
        for r in seats[seats.image == name].itertuples():
            p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
            color = (0, 0, 220) if r.gt_occ else (0, 180, 0)
            cv2.rectangle(img, p1, p2, color, 2)
            if isinstance(r.error, str) and r.error:
                cv2.rectangle(img, p1, p2, (0, 220, 255), 4)
                label = r.error.split()[0] + f" ov={r.overlap:.2f}"
                cv2.putText(img, label, (p1[0], max(12, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
        fn = os.path.join(out_dir, os.path.basename(name))
        cv2.imwrite(fn, img)
        tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
        print(f"{err_imgs[name]} ghế sai  ->  {fn}")

    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), np.vstack(rows))
        print("Lưới tổng hợp:", os.path.join(out_dir, "contact_sheet.jpg"))

    # gợi ý nhóm nguyên nhân: điền cột 'cause' bằng tay rồi đếm
    cause_csv = os.path.join(args.out, f"error_causes_{args.tag}.csv")
    if not os.path.exists(cause_csv):
        pd.DataFrame({"image": err_imgs.index, "n_wrong": err_imgs.values,
                      "cause": ""}).to_csv(cause_csv, index=False)
        print("Điền nguyên nhân (che khuất / góc nhìn / balo / ánh sáng / nhãn sai) vào:", cause_csv)


if __name__ == "__main__":
    main()
