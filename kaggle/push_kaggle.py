"""Đẩy LibSeat lên GPU Kaggle (T4) và mở demo qua ngrok.

Đọc KAGGLE_API_TOKEN, NGROK_AUTHTOKEN, NGROK_DOMAIN từ .env ở gốc repo. Token ngrok được đưa
lên Kaggle dưới dạng dataset riêng tư <user>/libseat-secrets, không nằm trong code của kernel.

    python kaggle/push_kaggle.py               # cập nhật dataset bí mật, chạy kernel, chờ server lên
    python kaggle/push_kaggle.py --hours 1     # thời gian giữ server (mặc định 2 giờ, tối đa 11)
    python kaggle/push_kaggle.py --videos ""   # không gửi video_test/ (mặc định gửi nếu có .mp4)
    python kaggle/push_kaggle.py --keep-videos # dùng lại dataset video đã gửi lần trước
    python kaggle/push_kaggle.py status        # trạng thái kernel
    python kaggle/push_kaggle.py output        # tải outputs/ của kernel về kaggle/output/
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
KERNEL_SLUG = "libseat-gpu-demo"
KERNEL_TITLE = "LibSeat GPU Demo"
SECRETS_SLUG = "libseat-secrets"
VIDEOS_SLUG = "libseat-videos"
ACCELERATOR = "NvidiaTeslaT4"


def load_env(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def push_dataset(api, ref, title, fill):
    """Tạo hoặc cập nhật dataset riêng tư; fill(folder) ghi file dữ liệu vào folder."""
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "dataset-metadata.json").write_text(json.dumps(
            {"title": title, "id": ref, "licenses": [{"name": "other"}]}), encoding="utf-8")
        fill(folder)
        try:
            api.dataset_status(ref)
            result = api.dataset_create_version(folder, "Cập nhật", quiet=True, delete_old_versions=True)
        except Exception:
            result = api.dataset_create_new(folder, public=False, quiet=True)
    if getattr(result, "error", None):
        sys.exit(f"Lỗi dataset {ref}: {result.error}")
    print(f"Dataset riêng tư {ref}: đã cập nhật, chờ Kaggle xử lý…")
    for _ in range(120):
        try:
            if str(api.dataset_status(ref)).lower().endswith("ready"):
                return ref
        except Exception:  # dataset vừa tạo có thể trả 403/404 trong lúc Kaggle xử lý
            pass
        time.sleep(5)
    sys.exit(f"Dataset {ref} chưa sẵn sàng sau 10 phút, thử lại sau.")


def push_secrets(api, user):
    def fill(folder):
        Path(folder, "libseat_secrets.json").write_text(json.dumps(
            {"ngrok_authtoken": os.environ["NGROK_AUTHTOKEN"],
             "ngrok_domain": os.environ.get("NGROK_DOMAIN", "")}), encoding="utf-8")
    return push_dataset(api, f"{user}/{SECRETS_SLUG}", "LibSeat secrets", fill)


def push_videos(api, user, videos):
    def fill(folder):
        for path in videos:  # tên không dấu, không khoảng trắng để Kaggle và OpenCV đọc được
            name = unicodedata.normalize("NFKD", path.stem.replace("đ", "d").replace("Đ", "D"))
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.encode("ascii", "ignore").decode()).strip("_")[:80]
            shutil.copy2(path, Path(folder, (name or "video") + path.suffix.lower()))
    print(f"Gửi {len(videos)} video test ({sum(p.stat().st_size for p in videos) / 1e6:.0f} MB)…")
    return push_dataset(api, f"{user}/{VIDEOS_SLUG}", "LibSeat test videos", fill)


def push_kernel(api, user, sources, hours):
    source = (HERE / "libseat_kaggle.py").read_text(encoding="utf-8")
    source = re.sub(r"^SERVE_HOURS = .*$", f"SERVE_HOURS = {hours}", source, count=1, flags=re.M)
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "libseat_kaggle.py").write_text(source, encoding="utf-8")
        Path(folder, "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_TITLE, "code_file": "libseat_kaggle.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": sources,
            "competition_sources": [], "kernel_sources": []}), encoding="utf-8")
        # Dư 30 phút để clone, cài thư viện và lưu output; chặn kernel treo đốt quota GPU.
        result = api.kernels_push(folder, timeout=str(int(hours * 3600 + 1800)), acc=ACCELERATOR)
    if getattr(result, "error", None):
        sys.exit(f"Lỗi đẩy kernel: {result.error}")
    version = getattr(result, "version_number", None) or getattr(result, "versionNumber", "?")
    print(f"Kernel: {result.url} (phiên bản {version})")


def kernel_status(api, ref):
    status = api.kernels_status(ref)
    state = str(getattr(status, "status", status)).split(".")[-1].lower()
    return state, getattr(status, "failure_message", "") or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="push", choices=["push", "status", "output"])
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--videos", default=str(ROOT / "video_test"), help="thư mục video test của nhóm")
    parser.add_argument("--keep-videos", action="store_true", help="dùng lại dataset video đã gửi")
    args = parser.parse_args()
    if not 0 < args.hours <= 11:
        sys.exit("--hours phải trong khoảng (0, 11]: Kaggle giới hạn 12 giờ mỗi phiên.")

    load_env(ROOT / ".env")  # phải trước khi import kaggle vì kaggle xác thực ngay khi import
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    ref = f"{user}/{KERNEL_SLUG}"

    if args.command == "status":
        state, message = kernel_status(api, ref)
        print(f"{ref}: {state} {message}")
        return
    if args.command == "output":
        out = HERE / "output"
        out.mkdir(exist_ok=True)
        files, _ = api.kernels_output(ref, str(out), force=True, page_size=100)
        print(f"Đã tải {len(files)} file vào {out}")
        return

    sources = [push_secrets(api, user)]
    videos = sorted(Path(args.videos).glob("*.mp4")) if args.videos else []
    if args.keep_videos:
        sources.append(f"{user}/{VIDEOS_SLUG}")
    elif videos:
        sources.append(push_videos(api, user, videos))
    push_kernel(api, user, sources, args.hours)
    domain = os.environ.get("NGROK_DOMAIN", "")
    print("Chờ kernel chạy (clone, cài thư viện, tải YOLO11s mất khoảng 2-4 phút)…")
    for _ in range(60):
        state, message = kernel_status(api, ref)
        if state not in ("queued", "running"):
            sys.exit(f"Kernel dừng: {state} {message}. Xem log: https://www.kaggle.com/code/{ref}")
        if state == "running":
            break
        time.sleep(10)
    print(f"Kernel đang chạy. Mở https://{domain} sau 2-4 phút (ngrok có thể hỏi xác nhận lần đầu).")
    print("Tắt sớm bằng nút trên trang để tiết kiệm quota GPU; kết quả: python kaggle/push_kaggle.py output")


if __name__ == "__main__":
    main()
