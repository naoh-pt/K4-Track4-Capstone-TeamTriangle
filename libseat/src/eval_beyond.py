"""[C] Đánh giá trạng thái từng ghế trên dataset có nhãn (Beyond, seatanalysis...).

Ý tưởng: dùng hộp ghế của NHÃN làm vùng ghế, rồi xem YOLO có phát hiện người
chồng vào vùng đó không. Như vậy số đo không phụ thuộc YOLO có thấy ghế hay không.

Hai luật gán được so sánh (bảng trước/sau):
  - overlap : chỉ tỉ lệ diện tích ghế bị hộp người che (bản thô, giống IoU trên ảnh)
  - foot    : overlap + điểm chân người (giữa cạnh dưới hộp) phải nằm gần ghế.
              Đây là phiên bản trên ảnh tĩnh của luật "khoảng cách sàn" của vai B:
              ảnh dataset không có 4 mốc nên không tính được homography thật.

Ngưỡng được chỉnh trên tập CALIB (một nửa ảnh), số cuối báo cáo trên tập TEST
(nửa còn lại) để con số không bị "học thuộc".

Chạy:
    python src/eval_beyond.py --data data/beyond
    python src/eval_beyond.py --data data/seatanalysis --tag seatanalysis
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OCC_WORDS = ("occupied", "occ", "taken", "full")
EMPTY_WORDS = ("empty", "free", "vacant", "available")


# ---------------------------------------------------------------- dữ liệu
def load_class_map(data_dir):
    """Đọc data.yaml, trả về {class_id: 'occ'|'empty'} cho các lớp ghế."""
    with open(os.path.join(data_dir, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cmap = {}
    for i, n in enumerate(names):
        low = n.lower()
        if "chair" not in low and "seat" not in low:
            continue  # bỏ qua lớp không phải ghế (vd standing-person, bag)
        if any(w in low for w in OCC_WORDS):
            cmap[i] = "occ"
        elif any(w in low for w in EMPTY_WORDS):
            cmap[i] = "empty"
    if not cmap:
        raise SystemExit(f"Không tìm thấy lớp ghế trống/có người trong {names}")
    print("Lớp trong data.yaml:", names)
    print("Ánh xạ dùng để đo:", {names[k]: v for k, v in cmap.items()})
    return names, cmap


def list_images(data_dir):
    """Gom ảnh của mọi split (train/valid/test). Không train nên dùng hết để đo."""
    out = []
    for img in sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)):
        if not img.lower().endswith(IMG_EXT) or f"{os.sep}images{os.sep}" not in img:
            continue
        lbl = img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        lbl = os.path.splitext(lbl)[0] + ".txt"
        if os.path.exists(lbl):
            out.append((img, lbl))
    return out


def split_of(path):
    """Chia CALIB/TEST cố định theo hash tên file (chạy lại vẫn ra cùng kết quả)."""
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    return "calib" if h % 2 == 0 else "test"


def read_seats(lbl, w, h, cmap):
    seats = []
    for line in open(lbl, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        if c not in cmap:
            continue
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:  # bbox: x y w h
            x, y, bw, bh = vals
            box = ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
        else:  # polygon: x1 y1 x2 y2 ... -> bbox bao ngoài
            xs, ys = np.array(vals[0::2]) * w, np.array(vals[1::2]) * h
            box = (xs.min(), ys.min(), xs.max(), ys.max())
        seats.append((cmap[c] == "occ", box))
    return seats


# ---------------------------------------------------------------- luật gán
def overlap_ratio(seat, person):
    sx1, sy1, sx2, sy2 = seat
    px1, py1, px2, py2 = person
    iw = max(0.0, min(sx2, px2) - max(sx1, px1))
    ih = max(0.0, min(sy2, py2) - max(sy1, py1))
    return (iw * ih) / max(1e-6, (sx2 - sx1) * (sy2 - sy1))


def foot_dist(seat, person):
    """Khoảng cách từ chân người tới tâm-dưới ghế, chuẩn hóa theo chiều cao ghế."""
    sx1, sy1, sx2, sy2 = seat
    px1, _, px2, py2 = person
    foot = np.array([(px1 + px2) / 2, py2])
    seat_base = np.array([(sx1 + sx2) / 2, sy2])
    return float(np.linalg.norm(foot - seat_base) / max(1e-6, sy2 - sy1))


def features(seat, persons):
    """Với mỗi ghế: (overlap lớn nhất, khoảng cách chân của người có overlap đó)."""
    best_ov, best_fd = 0.0, np.inf
    for p in persons:
        ov = overlap_ratio(seat, p)
        if ov > best_ov:
            best_ov, best_fd = ov, foot_dist(seat, p)
    return best_ov, best_fd


def predict(df, rule, ov_thr, fd_thr=None):
    if rule == "overlap":
        return df.overlap > ov_thr
    return (df.overlap > ov_thr) & (df.foot_dist < fd_thr)


def metrics(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    tp = int((gt & pred).sum()); tn = int((~gt & ~pred).sum())
    fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    n = len(gt)
    acc = (tp + tn) / max(1, n)
    # balanced accuracy: công bằng khi số ghế trống/có người lệch nhau
    tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)
    return dict(n=n, acc=acc, bal_acc=(tpr + tnr) / 2, tp=tp, tn=tn, fp=fp, fn=fn)


# ---------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/beyond")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", default="beyond")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from ultralytics import YOLO

    names, cmap = load_class_map(args.data)
    pairs = list_images(args.data)
    if not pairs:
        raise SystemExit(f"Không có ảnh + nhãn trong {args.data}/**/images")
    model = YOLO(args.model)
    assert model.names[0] == "person", model.names[0]

    rows, t_infer = [], []
    for img, lbl in pairs:
        t0 = time.perf_counter()
        res = model(img, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        t_infer.append(time.perf_counter() - t0)
        h, w = res.orig_shape
        persons = res.boxes.xyxy.cpu().numpy().tolist()
        for k, (gt_occ, box) in enumerate(read_seats(lbl, w, h, cmap)):
            ov, fd = features(box, persons)
            rows.append(dict(image=os.path.relpath(img, args.data), seat=k,
                             split=split_of(img), gt_occ=gt_occ, overlap=ov,
                             foot_dist=fd, n_persons=len(persons),
                             x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    df = pd.DataFrame(rows)
    cal, test = df[df.split == "calib"], df[df.split == "test"]

    # --- chỉnh ngưỡng trên CALIB (tối đa balanced accuracy)
    ov_grid = np.round(np.arange(0.05, 0.80, 0.05), 2)
    fd_grid = np.round(np.arange(0.25, 3.01, 0.25), 2)
    best = {}
    best["overlap"] = max(((metrics(cal.gt_occ, predict(cal, "overlap", o))["bal_acc"], o, None)
                           for o in ov_grid))
    best["foot"] = max(((metrics(cal.gt_occ, predict(cal, "foot", o, f))["bal_acc"], o, f)
                        for o in ov_grid for f in fd_grid))
    # bản "mặc định của plan" (ngưỡng 0.3, không chỉnh) làm điểm xuất phát thật thà
    configs = [("Chỉ YOLO + overlap, ngưỡng mặc định 0.30", "overlap", 0.30, None),
               ("Chỉ YOLO + overlap, ngưỡng đã chỉnh", "overlap", best["overlap"][1], None),
               ("+ Luật điểm chân (gần đúng hình học sàn)", "foot", best["foot"][1], best["foot"][2])]

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for name, rule, o, f in configs:
        m = metrics(test.gt_occ, predict(test, rule, o, f))
        mc = metrics(cal.gt_occ, predict(cal, rule, o, f))
        summary.append(dict(version=name, rule=rule, ov_thr=o, fd_thr=f,
                            calib_bal_acc=mc["bal_acc"], **m))
    sm = pd.DataFrame(summary)

    # --- chọn phiên bản cuối theo điểm CALIB (không nhìn TEST), lưu ca lỗi của nó
    final = configs[-1] if best["foot"][0] >= best["overlap"][0] else configs[1]
    df["pred_occ"] = predict(df, final[1], final[2], final[3])
    df["error"] = np.where(df.gt_occ == df.pred_occ, "",
                           np.where(df.pred_occ, "FP (trống bị báo có người)",
                                    "FN (có người bị báo trống)"))
    df.to_csv(os.path.join(args.out, f"seats_{args.tag}.csv"), index=False)
    df[df.error != ""].to_csv(os.path.join(args.out, f"errors_{args.tag}.csv"), index=False)

    info = dict(dataset=args.data, tag=args.tag, model=args.model, conf=args.conf,
                imgsz=args.imgsz, n_images=len(pairs), n_seats=len(df),
                n_calib=len(cal), n_test=len(test),
                occ_share=float(df.gt_occ.mean()),
                infer_ms_mean=1000 * float(np.mean(t_infer)),
                final_version=final[0], final_ov_thr=final[2], final_fd_thr=final[3],
                class_names=names)
    with open(os.path.join(args.out, f"run_{args.tag}.json"), "w", encoding="utf-8") as fp:
        json.dump(dict(info=info, summary=summary), fp, ensure_ascii=False, indent=2, default=str)

    # --- in và ghi metrics.md
    fin = sm.iloc[-1] if final is configs[-1] else sm.iloc[1]
    lines = [f"## Kết quả trên `{args.tag}` ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"- Dữ liệu: `{args.data}`, {len(pairs)} ảnh, {len(df)} ghế có nhãn "
             f"({df.gt_occ.mean():.0%} có người). CALIB {len(cal)} ghế / TEST {len(test)} ghế.",
             f"- Model: `{args.model}`, conf={args.conf}, imgsz={args.imgsz}, "
             f"suy luận TB {info['infer_ms_mean']:.0f} ms/ảnh trên máy đo.",
             "- Ngưỡng chỉnh trên CALIB, số dưới đây đo trên TEST.", "",
             "| Phiên bản | Độ chính xác | Balanced acc. | FP | FN | Ngưỡng |",
             "|---|---|---|---|---|---|"]
    for r in summary:
        thr = f"ov>{r['ov_thr']}" + (f", chân<{r['fd_thr']}" if r["fd_thr"] else "")
        lines.append(f"| {r['version']} | {r['acc']:.1%} | {r['bal_acc']:.1%} | "
                     f"{r['fp']} | {r['fn']} | {thr} |")
    lines += ["", f"**Số đo chính:** {fin.acc:.1%} độ chính xác từng ghế trên {int(fin.n)} ghế "
              f"(TEST, `{args.tag}`). Ma trận: TP={fin.tp}, TN={fin.tn}, FP={fin.fp}, FN={fin.fn}.", ""]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out, f"metrics_{args.tag}.md"), "w", encoding="utf-8") as fp:
        fp.write(text)


if __name__ == "__main__":
    main()
