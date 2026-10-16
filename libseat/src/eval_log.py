"""[C] Phân tích log CSV của demo: nhấp nháy trước/sau làm mượt, FPS, độ trễ.

Định dạng log (vai A ghi, mỗi dòng = 1 ghế trong 1 khung):
    t,frame,seat_id,raw_occ,smooth_occ
    1729070000.12,0,0,0,0
    ...
  t          : time.time() lúc xử lý khung
  raw_occ    : 0/1 trạng thái thô (trước hysteresis)
  smooth_occ : 0/1 trạng thái đã làm mượt (sau SeatState.update)

Độ trễ bấm tay (tùy chọn) - file CSV:
    trial,latency_s
    1,0.9
    ...

Chạy:
    python src/eval_log.py --log data/logs/run1.csv [--stopwatch data/logs/latency.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def changes(s):
    s = s.astype(int).to_numpy()
    return int((s[1:] != s[:-1]).sum())


def smoothing_delay(g):
    """Trễ do làm mượt: từ lúc raw bật lên (và giữ tới khi smooth bật) tới lúc smooth bật."""
    t, raw, sm = g.t.to_numpy(), g.raw_occ.to_numpy(), g.smooth_occ.to_numpy()
    out = []
    for i in range(1, len(sm)):
        if sm[i] != sm[i - 1]:
            j = i
            while j > 0 and raw[j - 1] == sm[i]:
                j -= 1
            out.append((("on" if sm[i] else "off"), t[i] - t[j]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--stopwatch")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.log).sort_values(["seat_id", "frame"])
    frames = df.drop_duplicates("frame").sort_values("frame")
    dur = frames.t.iloc[-1] - frames.t.iloc[0]
    fps = (len(frames) - 1) / max(dur, 1e-6)
    minutes = dur / 60

    per = []
    delays = []
    for sid, g in df.groupby("seat_id"):
        per.append(dict(seat=sid, raw=changes(g.raw_occ), smooth=changes(g.smooth_occ)))
        delays += smoothing_delay(g)
    per = pd.DataFrame(per)
    raw_c, sm_c = per.raw.sum(), per.smooth.sum()

    lines = [f"## Log demo `{os.path.basename(args.log)}`", "",
             f"- {len(frames)} khung trong {dur:.0f} s, **FPS trung bình {fps:.1f}**, "
             f"{df.seat_id.nunique()} ghế.",
             f"- Số lần đổi trạng thái (mọi ghế): thô **{raw_c}** -> đã làm mượt **{sm_c}** "
             f"({raw_c / max(minutes, 1e-6):.1f} -> {sm_c / max(minutes, 1e-6):.1f} lần/phút).",
             ]
    if delays:
        d = pd.DataFrame(delays, columns=["kind", "s"])
        for kind, g in d.groupby("kind"):
            word = "ngồi xuống -> đỏ" if kind == "on" else "đứng dậy -> xanh"
            lines.append(f"- Trễ do làm mượt ({word}): TB {g.s.mean():.2f} s, "
                         f"độ lệch {g.s.std(ddof=0):.2f} s, n={len(g)}.")
    if args.stopwatch:
        sw = pd.read_csv(args.stopwatch)
        lines.append(f"- Trễ bấm tay ngồi xuống -> đổi màu: TB {sw.latency_s.mean():.2f} s, "
                     f"độ lệch {sw.latency_s.std(ddof=1):.2f} s, n={len(sw)} lần.")
    lines += ["", "| Ghế | Đổi trạng thái thô | Sau làm mượt |", "|---|---|---|"]
    lines += [f"| {r.seat} | {r.raw} | {r.smooth} |" for r in per.itertuples()]
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.log))[0]
    with open(os.path.join(args.out, f"metrics_log_{name}.md"), "w", encoding="utf-8") as fp:
        fp.write(text)


if __name__ == "__main__":
    main()
