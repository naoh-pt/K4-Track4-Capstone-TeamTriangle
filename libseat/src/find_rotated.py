"""[C] Tìm ảnh bị xoay 90° trong dataset: YOLO thấy nhiều người hơn hẳn khi xoay ảnh lại.

Chạy: python src/find_rotated.py --data data/beyond --tag beyond
Ra:   results/rotated_<tag>.csv (image, n0, n90, n270, rotated)
"""
import argparse
import glob
import os
import sys

import cv2
import pandas as pd


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/beyond")
    ap.add_argument("--tag", default="beyond")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")

    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*", "images", "*.jpg"))):
        img = cv2.imread(p)
        n = {}
        for k, rot in [(0, None), (90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            im = img if rot is None else cv2.rotate(img, rot)
            n[k] = len(model(im, classes=[0], conf=0.25, verbose=False)[0].boxes)
        rotated = max(n[90], n[270]) >= n[0] + 2   # thấy thêm >= 2 người khi xoay
        rows.append(dict(image=os.path.relpath(p, args.data), n0=n[0], n90=n[90], n270=n[270], rotated=rotated))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"rotated_{args.tag}.csv"), index=False)
    print(f"{df.rotated.sum()}/{len(df)} ảnh có dấu hiệu bị xoay 90°")


if __name__ == "__main__":
    main()
