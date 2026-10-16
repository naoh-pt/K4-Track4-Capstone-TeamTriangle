"""[C] Ước lượng tác động từ khảo sát nhanh trong lớp.

File khảo sát CSV (mỗi dòng = 1 người trả lời):
    minutes_search,visits_per_day,would_use
    6,1,1
    10,2,1
    ...
  minutes_search : số phút trung bình mất để tìm chỗ trống mỗi lần
  visits_per_day : số lần vào thư viện mỗi ngày (ngày có đến thư viện)
  would_use      : 1 nếu sẽ xem LibSeat trước khi đi, 0 nếu không

Chạy:
    python src/impact.py --survey data/survey.csv --users 500 --saved-share 0.6
  --users       : số SV dùng thư viện/ngày (GIẢ ĐỊNH, ghi rõ nguồn khi trình bày)
  --saved-share : tỉ lệ thời gian tìm chỗ hệ thống cắt được (GIẢ ĐỊNH, mặc định 0.6:
                  vẫn còn thời gian đi tới ghế)
"""
import argparse
import sys

import numpy as np
import pandas as pd


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--users", type=float, default=500)
    ap.add_argument("--saved-share", type=float, default=0.6)
    ap.add_argument("--days", type=int, default=150, help="số ngày học/năm")
    args = ap.parse_args()

    s = pd.read_csv(args.survey)
    n = len(s)
    med_min = s.minutes_search.median()
    mean_min = s.minutes_search.mean()
    visits = s.visits_per_day.mean()
    adopt = s.would_use.mean() if "would_use" in s else 1.0

    # bootstrap khoảng 90% cho số phút trung bình (mẫu nhỏ nên phải nói rõ độ bất định)
    rng = np.random.default_rng(0)
    boots = [rng.choice(s.minutes_search, n).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [5, 95])

    def hours(m):
        return args.users * adopt * visits * m * args.saved_share / 60

    print(f"Khảo sát: n={n} (mẫu nhỏ, chỉ mang tính định hướng)")
    print(f"  Phút tìm chỗ / lần: TB {mean_min:.1f} (khoảng 90%: {lo:.1f}-{hi:.1f}), trung vị {med_min:.1f}")
    print(f"  Lần vào thư viện / ngày: {visits:.2f}   |   Sẽ dùng: {adopt:.0%}")
    print()
    print("Công thức: SV/ngày x tỉ lệ dùng x lần/ngày x phút tìm x tỉ lệ cắt được")
    print(f"  = {args.users:.0f} x {adopt:.2f} x {visits:.2f} x {mean_min:.1f} x {args.saved_share:.2f}")
    print(f"  ≈ {hours(mean_min):.0f} giờ SV / ngày  (khoảng {hours(lo):.0f}-{hours(hi):.0f})")
    print(f"  ≈ {hours(mean_min) * args.days:,.0f} giờ SV / năm học ({args.days} ngày)")
    print(f"  Mỗi SV dùng: tiết kiệm ≈ {visits * mean_min * args.saved_share:.1f} phút / ngày")


if __name__ == "__main__":
    main()
