"""[C] So các cấu hình bộ phát hiện (model, imgsz, conf). Chọn theo điểm CALIB.

Chạy:
    python src/sweep_detector.py --data data/beyond --tag beyond
Ra: results/sweep_<tag>.md
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

CONFIGS = [("yolo11n.pt", 640, 0.25), ("yolo11n.pt", 640, 0.10),
           ("yolo11n.pt", 960, 0.10), ("yolo11n.pt", 1280, 0.10),
           ("yolo11s.pt", 960, 0.10), ("yolo11s.pt", 1280, 0.10)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/beyond")
    ap.add_argument("--tag", default="beyond")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for model, imgsz, conf in CONFIGS:
        sub = f"{args.tag}_{os.path.splitext(model)[0]}_{imgsz}_c{int(conf * 100)}"
        subprocess.run([sys.executable, os.path.join(here, "eval_beyond.py"), "--data", args.data,
                        "--tag", sub, "--model", model, "--imgsz", str(imgsz), "--conf", str(conf),
                        "--out", os.path.join(args.out, "sweep")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run = json.load(open(os.path.join(args.out, "sweep", f"run_{sub}.json"), encoding="utf-8"))
        best = max(run["summary"], key=lambda r: r["calib_bal_acc"])
        rows.append(dict(model=model, imgsz=imgsz, conf=conf, ms=run["info"]["infer_ms_mean"], **best))
        print(f"{model} {imgsz} conf={conf}: calib bal {best['calib_bal_acc']:.3f} | "
              f"test acc {best['acc']:.3f} bal {best['bal_acc']:.3f} FN {best['fn']} FP {best['fp']}")

    pick = max(rows, key=lambda r: r["calib_bal_acc"])
    lines = [f"## So cấu hình bộ phát hiện trên `{args.tag}` (chọn theo CALIB)", "",
             "| Model | imgsz | conf | ms/ảnh | Luật | CALIB bal. | TEST acc. | TEST bal. | FP | FN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " **(chọn)**" if r is pick else ""
        lines.append(f"| {r['model']}{mark} | {r['imgsz']} | {r['conf']} | {r['ms']:.0f} | {r['rule']} | "
                     f"{r['calib_bal_acc']:.1%} | {r['acc']:.1%} | {r['bal_acc']:.1%} | {r['fp']} | {r['fn']} |")
    with open(os.path.join(args.out, f"sweep_{args.tag}.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
