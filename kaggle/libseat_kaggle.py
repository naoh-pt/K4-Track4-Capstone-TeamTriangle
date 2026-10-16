"""LibSeat trên GPU Kaggle: demo webcam qua ngrok, video test và video demo sản phẩm.

Chạy như một Kaggle script (đẩy bằng kaggle/push_kaggle.py):
1. Clone repo, cài ultralytics + ngrok, tải YOLO11s (cấu hình vai C: imgsz 960, conf 0.10) và YOLO11s-pose.
2. Video test: video của nhóm (dataset riêng tư libseat-videos), demo_backup.mp4 và 2 clip Pexels.
3. Dựng video demo sản phẩm libseat_demo.mp4: thẻ giới thiệu, pipeline, clip đã chạy, số đo của vai C.
4. Mở server web qua ngrok (domain tĩnh): trình duyệt gửi khung webcam lên GPU, nhận lại khung
   đã vẽ; quay video test, dựng lại video demo từ bản quay, tải log CSV.

Chống nhầm và chống "quên rồi gán mới":
- Người: BoT-SORT (bù chuyển động camera) giữ ID khi bị che tới TRACK_SECONDS; hộp yếu (0,10-0,35) chỉ nối
  track cũ, track mới cần >= NEW_TRACK_CONF và phải xuất hiện CONFIRM_FRAMES khung mới được tính.
- Tư thế: YOLO11s-pose, bỏ phiếu theo khung; người đang đứng không chiếm ghế.
- Ghế: SeatRegistry giữ ID cố định, bám theo chuyển động camera; ghế bị che không bị quên; ghế mới chỉ thêm khi
  thấy liên tục SEAT_NEW_SECONDS; người ngồi yên SIT_INFER_SECONDS ở chỗ chưa có ghế thì suy ra ghế.
Gán người vào ghế (assign_people) và làm mượt (SeatState) dùng code của vai B (src/geometry.py, src/smoothing.py).
Bí mật ngrok đọc từ dataset riêng tư libseat-secrets (libseat_secrets.json), không nằm trong code.
Mọi kết quả nằm trong /kaggle/working/outputs/ (Output của kernel).
"""
import collections
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

SERVE_HOURS = 2.0  # push_kaggle.py --hours ghi đè dòng này
REPO_URL = "https://github.com/naoh-pt/K4-Track4-Capstone-TeamTriangle"
WORK_DIR = "/tmp/libseat"
REPO_DIR = WORK_DIR + "/repo"
OUT_DIR = "/kaggle/working/outputs"
PORT = 8000

MODEL = "yolo11s.pt"
POSE_MODEL = "yolo11s-pose.pt"
IMGSZ = 960
CONF = 0.10        # ngưỡng YOLO, vai C chọn trên CALIB (libseat/results/metrics.md)
SEAT_CONF = 0.35   # chỉ hộp chair đủ tin cậy mới thành ghế
OVERLAP = 0.3
ON_S, OFF_S = 0.5, 1.0
GAP_VIDEO = 0.5    # mặc định của vai B
# Webcam qua ngrok tới Kaggle: đo được khứ hồi ~0,6 s, mỗi khung ~1,1-1,5 s. Khoảng cách khung lớn hơn
# max_gap sẽ xóa bằng chứng đang chờ và ghế không bao giờ đổi trạng thái, nên nới có chủ đích.
GAP_WEBCAM = 2.5
WEBCAM_FPS = 3     # ước lượng tốc độ khung webcam qua ngrok, dùng cho tracker
REC_FPS = 15
PERSON, CHAIR = 0, 56
MAX_SIDE = 1920          # khung 4K được thu về cạnh dài 1920 px trước khi xử lý
MIN_SIDE = 1280          # khung nhỏ (CCTV 640x352) được phóng lên để bảng đếm không che ghế
MIN_PERSON_H = 0.02      # bỏ hộp người thấp hơn 2% cạnh ngắn của khung (ảnh người trên poster, nhiễu)
NEW_TRACK_CONF = 0.35    # track người mới phải có hộp >= 0,35
CONFIRM_FRAMES = 3       # người phải xuất hiện 3 khung mới được tính
TRACK_SECONDS = 3.0      # giữ ID người bị che tối đa 3 s
POSTURE_WINDOW = 15      # bỏ phiếu tư thế trên 15 khung gần nhất
SEAT_MATCH_IOU = 0.3
SEAT_NEW_SECONDS = 0.5   # ghế mới phải thấy liên tục 0,5 s
SIT_INFER_SECONDS = 1.5  # ngồi yên 1,5 s ở chỗ chưa có ghế thì suy ra ghế
SEAT_VISIBLE = 0.6       # ghế có >= 60% diện tích trong khung mới coi là trong khung

# Clip Pexels (giấy phép Pexels, miễn phí): (tên, URL, số giây dùng, mô tả cảnh trong video demo)
FOOTAGE = [
    ("pexels_9883925", "https://videos.pexels.com/video-files/9883925/9883925-hd_1280_720_30fps.mp4", 20,
     "Văn phòng, camera cố định: một người bước vào và ngồi xuống"),
    ("pexels_8499766", "https://videos.pexels.com/video-files/8499766/8499766-hd_1280_720_30fps.mp4", 26,
     "Lớp học, camera lia chậm: học sinh đi ngang qua"),
]
DEMO_FPS, DEMO_SIZE, CAPTION_H = 25, (1280, 720), 112


def sh(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


sh(sys.executable, "-m", "pip", "install", "-q", "ultralytics", "ngrok", "flask")
if not os.path.isdir(REPO_DIR):
    sh("git", "clone", "--depth", "1", REPO_URL, REPO_DIR)
sys.path.insert(0, REPO_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(WORK_DIR)  # để file .pt tải về đây, không lọt vào output Kaggle

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ultralytics  # noqa: E402
import yaml  # noqa: E402
from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.trackers.bot_sort import BOTSORT  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from src.geometry import Person, Seat, assign_people, overlap_ratio  # noqa: E402
from src.smoothing import SeatState  # noqa: E402

DEVICE = 0 if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
det_model = YOLO(MODEL)
pose_model = YOLO(POSE_MODEL)
assert det_model.names[PERSON] == "person" and det_model.names[CHAIR] == "chair", det_model.names
model_lock = threading.Lock()
stop_event = threading.Event()
started = time.time()
print(f">>> {MODEL} + {POSE_MODEL} trên {GPU_NAME}, ultralytics {ultralytics.__version__}", flush=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, GRAY = (0, 200, 0), (0, 0, 255), (160, 160, 160)
POSTURE_COLOR = {"sitting": (0, 165, 255), "standing": (255, 200, 0), "unknown": (210, 210, 210)}
POSTURE_TEXT = {"sitting": "sit", "standing": "stand", "unknown": "?"}


# ---------- Hình học nhỏ ----------

def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def warp_box(box, m):
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float) @ m[:, :2].T + m[:, 2]
    return np.array([*corners.min(0), *corners.max(0)])


def seat_region(box):
    """Vùng mông/đùi của người ngồi: nửa dưới hộp người, thu hẹp 2 bên."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + 0.15 * w, y1 + 0.5 * h, x2 - 0.15 * w, y2])


def fit(frame):
    h, w = frame.shape[:2]
    scale = MAX_SIDE / max(h, w) if max(h, w) > MAX_SIDE else max(1.0, MIN_SIDE / max(h, w))
    if scale == 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


# ---------- Phát hiện, tư thế, chuyển động camera, tracker ----------

def detect(frame):
    """Người + ghế từ YOLO11s (đúng cấu hình vai C đo); khớp tay chân từ YOLO11s-pose, ghép theo IoU."""
    with model_lock:
        t0 = time.perf_counter()
        r = det_model.predict(frame, imgsz=IMGSZ, conf=CONF, classes=[PERSON, CHAIR],
                              device=DEVICE, verbose=False)[0]
        rp = pose_model.predict(frame, imgsz=IMGSZ, conf=0.25, device=DEVICE, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
    boxes = r.boxes.cpu().numpy()
    xyxy = boxes.xyxy
    ok = (xyxy[:, 2] > xyxy[:, 0] + 1) & (xyxy[:, 3] > xyxy[:, 1] + 1)
    tall = (xyxy[:, 3] - xyxy[:, 1]) >= MIN_PERSON_H * min(frame.shape[:2])  # video dọc: không loại người ở xa
    people = boxes[(boxes.cls == PERSON) & ok & tall]
    chair_mask = (boxes.cls == CHAIR) & ok
    chairs = [(np.array(b), float(c)) for b, c in zip(xyxy[chair_mask], boxes.conf[chair_mask])]
    kpts = [None] * len(people)
    if len(people) and rp.keypoints is not None and len(rp.boxes):
        iou = box_iou(people.xyxy, rp.boxes.xyxy.cpu().numpy())
        data = rp.keypoints.data.cpu().numpy()
        for i, j in enumerate(iou.argmax(1)):
            if iou[i, j] >= 0.5:
                kpts[i] = data[j]
    return people, kpts, chairs, ms


def posture(kpts):
    """Ngồi/đứng từ khớp hông-gối-cổ chân (COCO 17 điểm). Không thấy hông hoặc gối thì 'unknown'."""
    if kpts is None or kpts.shape[-1] < 3:
        return "unknown"
    seen = kpts[:, 2] >= 0.5

    def mid(a, b):
        pts = [kpts[i, :2] for i in (a, b) if seen[i]]
        return np.mean(pts, 0) if pts else None

    shoulder, hip, knee, ankle = mid(5, 6), mid(11, 12), mid(13, 14), mid(15, 16)
    if hip is None or knee is None:
        return "unknown"
    torso = float(np.linalg.norm(shoulder - hip)) if shoulder is not None else 0.0
    thigh_dy = float(knee[1] - hip[1])
    if ankle is not None:
        a, b = hip - knee, ankle - knee
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if angle < 135:
            return "sitting"
        if angle > 160 and (not torso or thigh_dy > 0.6 * torso):
            return "standing"
    if torso:
        if thigh_dy < 0.45 * torso:
            return "sitting"
        if thigh_dy > 0.75 * torso:
            return "standing"
    return "unknown"


class CameraMotion:
    """Ước lượng chuyển động camera giữa 2 khung (optical flow trên nền, bỏ vùng có người)."""

    def __init__(self):
        self.prev = self.prev_mask = None

    def update(self, frame, person_boxes):
        h, w = frame.shape[:2]
        s = 640 / max(h, w)
        gray = cv2.cvtColor(cv2.resize(frame, (round(w * s), round(h * s))), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in np.asarray(person_boxes).reshape(-1, 4) * s:
            mask[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = 0
        prev, prev_mask = self.prev, self.prev_mask
        self.prev, self.prev_mask = gray, mask
        if prev is None:
            return None
        pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=prev_mask)
        if pts is None or len(pts) < 12:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = status.ravel() == 1
        if good.sum() < 12:
            return None
        m, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None:
            return None
        m[:, 2] /= s
        return m


def make_tracker(fps):
    with open(os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "botsort.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(track_high_thresh=0.25, track_low_thresh=CONF, new_track_thresh=NEW_TRACK_CONF,
               track_buffer=max(1, int(TRACK_SECONDS * fps)), with_reid=False, device=DEVICE)
    args = IterableSimpleNamespace(**cfg)
    try:
        return BOTSORT(args=args)
    except TypeError:  # ultralytics cũ: track_buffer tính theo 30 fps
        return BOTSORT(args=args, frame_rate=30)


# ---------- Ghế có ID ổn định và trạng thái ghế ----------

class SeatRegistry:
    """Danh sách ghế có ID cố định: bám theo camera, không quên ghế bị che, chỉ thêm ghế mới khi chắc chắn."""

    def __init__(self):
        self.boxes, self.kind = {}, {}  # seat_id -> hộp [x1,y1,x2,y2]; "chair" hoặc "person" (suy ra từ người ngồi)
        self.next_id = 1
        self.candidates = []
        self.sit_timer = {}  # track_id -> (vùng ngồi, thời điểm bắt đầu ngồi yên)

    def add(self, box, kind):
        seat_id = f"S{self.next_id}"
        self.next_id += 1
        self.boxes[seat_id], self.kind[seat_id] = np.array(box, float), kind
        return seat_id

    def seed(self, chairs):
        for box, conf in chairs:
            if conf >= SEAT_CONF and not self.overlaps(box):
                self.add(box, "chair")

    def warp(self, m):
        if m is None:
            return
        self.boxes = {k: warp_box(b, m) for k, b in self.boxes.items()}
        for cand in self.candidates:
            cand["box"] = warp_box(cand["box"], m)
        self.sit_timer = {k: (warp_box(b, m), since) for k, (b, since) in self.sit_timer.items()}

    def visible(self, width, height):
        out = set()
        for seat_id, (x1, y1, x2, y2) in self.boxes.items():
            inside = max(0, min(x2, width) - max(x1, 0)) * max(0, min(y2, height) - max(y1, 0))
            if inside >= SEAT_VISIBLE * (x2 - x1) * (y2 - y1):
                out.add(seat_id)
        return out

    def overlaps(self, box, thr=0.2):
        if not self.boxes:
            return False
        arr = np.array(list(self.boxes.values()))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = (arr[:, 0] <= cx) & (cx <= arr[:, 2]) & (arr[:, 1] <= cy) & (cy <= arr[:, 3])
        return bool((box_iou([box], arr)[0] >= thr).any() or inside.any())

    def update(self, chairs, sitters, t):
        strong = [b for b, c in chairs if c >= SEAT_CONF]
        ids = list(self.boxes)
        unmatched = set(range(len(strong)))
        if ids and strong:  # 1) hộp chair khớp ghế cũ: giữ ID, chỉnh vị trí
            iou = box_iou(strong, [self.boxes[i] for i in ids])
            for r, c in zip(*linear_sum_assignment(-iou)):
                if iou[r, c] >= SEAT_MATCH_IOU:
                    seat_id = ids[c]
                    if self.kind[seat_id] == "person":  # ghế suy ra nay đã thấy chair thật
                        self.boxes[seat_id], self.kind[seat_id] = strong[r], "chair"
                    else:
                        self.boxes[seat_id] = 0.7 * self.boxes[seat_id] + 0.3 * strong[r]
                    unmatched.discard(r)
        for r in unmatched:  # 2) chair mới: thành ứng viên, đủ bằng chứng mới thêm
            box = strong[r]
            if self.overlaps(box):
                continue
            match = next((c for c in self.candidates if box_iou([box], [c["box"]])[0, 0] >= 0.3), None)
            if match:
                match.update(box=0.5 * match["box"] + 0.5 * box, last=t, hits=match["hits"] + 1)
            else:
                self.candidates.append({"box": box, "first": t, "last": t, "hits": 1})
        keep = []
        for cand in self.candidates:
            if cand["hits"] >= 3 and cand["last"] - cand["first"] >= SEAT_NEW_SECONDS and not self.overlaps(cand["box"]):
                self.add(cand["box"], "chair")
            elif t - cand["last"] <= 1.0:
                keep.append(cand)
        self.candidates = keep
        for p in sitters:  # 3) ngồi yên ở chỗ chưa có ghế: ghế bị người che hoặc ghế băng
            region = seat_region(p["box"])
            if self.overlaps(region, thr=0.1):
                self.sit_timer.pop(p["id"], None)
                continue
            prev = self.sit_timer.get(p["id"])
            if prev is None or box_iou([region], [prev[0]])[0, 0] < 0.5:
                self.sit_timer[p["id"]] = (region, t)
            elif t - prev[1] >= SIT_INFER_SECONDS:
                self.add(region, "person")
                del self.sit_timer[p["id"]]
            else:
                self.sit_timer[p["id"]] = (0.7 * prev[0] + 0.3 * region, prev[1])
        alive = {p["id"] for p in sitters}
        self.sit_timer = {k: v for k, v in self.sit_timer.items() if k in alive}


class TrackedEngine:
    """Như OccupancyEngine của vai B (assign_people + SeatState), thêm 2 điều: ghế được thêm dần,
    và ghế ngoài khung nhận None (thiếu bằng chứng) nên giữ trạng thái cuối, không bị báo trống."""

    def __init__(self, max_gap):
        self.max_gap, self.states = max_gap, {}

    def update(self, registry, people, t, width, height):
        visible = registry.visible(width, height)
        seats = [Seat(i, tuple(float(v) for v in registry.boxes[i])) for i in registry.boxes if i in visible]
        matches = (assign_people(seats, people, None, overlap_threshold=OVERLAP, min_confidence=CONF)
                   if seats and people is not None else {})
        rows = {}
        for seat_id in registry.boxes:
            state = self.states.setdefault(seat_id, SeatState(ON_S, OFF_S, self.max_gap))
            seen = seat_id in visible and people is not None
            index = matches.get(seat_id) if seen else None
            raw = (index is not None) if seen else None
            rows[seat_id] = {"raw_occupied": raw, "occupied": state.update(raw, t), "person_index": index,
                             "visible": seat_id in visible, "kind": registry.kind[seat_id]}
        shown = [r for r in rows.values() if r["visible"]]
        return {"seats": rows, "total": len(rows), "visible": len(shown),
                "available": sum(r["occupied"] is False for r in shown),
                "occupied": sum(r["occupied"] is True for r in shown),
                "unknown": sum(r["occupied"] is None for r in shown),
                "out_of_view": len(rows) - len(shown),
                "chair_seats": sum(k == "chair" for k in registry.kind.values()),
                "inferred_seats": sum(k == "person" for k in registry.kind.values())}


# ---------- Vẽ ----------

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def font(size, bold=False):
    try:  # DejaVu có đủ dấu tiếng Việt
        return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except OSError:
        return ImageFont.load_default()


def label(img, text, org, color, scale=0.5, thick=1):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 2), FONT, scale, color, thick, cv2.LINE_AA)


def render_hud(status, counts, ms, u):
    """Bảng đếm (tiếng Việt có dấu) ở góc trên trái."""
    big, small, pad = int(25 * u), int(17 * u), int(14 * u)
    rows = [("Ghế trống", status["available"], (90, 210, 130)), ("Có người", status["occupied"], (255, 105, 105)),
            ("Chưa rõ", status["unknown"], (175, 180, 188)), ("Ngoài khung", status["out_of_view"], (140, 148, 158))]
    info = [f"Tổng ghế {status['total']}: {status['chair_seats']} từ chair, {status['inferred_seats']} suy ra",
            f"Người {counts['people']}: ngồi {counts['sitting']}, đứng {counts['standing']}, ? {counts['posture_unknown']}",
            f"ID người đã thấy: {counts['ids']}",
            f"{MODEL[:-3]} + pose · {GPU_NAME} · {ms:.0f} ms"]
    width = int(440 * u)
    height = pad * 2 + int(big * 1.3) + len(rows) * int(big * 1.3) + len(info) * int(small * 1.45) + pad
    img = Image.new("RGB", (width, height), (14, 16, 20))
    d = ImageDraw.Draw(img)
    y = pad
    d.text((pad, y), "LibSeat", font=font(big, True), fill=(90, 165, 255))
    y += int(big * 1.3)
    for name, value, color in rows:
        d.text((pad, y), name, font=font(big), fill=(230, 232, 236))
        text = str(value)
        d.text((width - pad - d.textlength(text, font=font(big, True)), y), text, font=font(big, True), fill=color)
        y += int(big * 1.3)
    d.line([(pad, y + 4), (width - pad, y + 4)], fill=(60, 66, 74), width=1)
    y += pad
    for line in info:
        d.text((pad, y), line, font=font(small), fill=(185, 190, 198))
        y += int(small * 1.45)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw(frame, registry, status, persons, sitters, counts, ms, recording):
    img = frame.copy()
    height, width = img.shape[:2]
    u = max(0.75, min(width, height) / 1080)  # khung webcam 960x540 vẫn đọc được bảng đếm
    crowded = len(persons) > 40
    for p in persons:
        x1, y1, x2, y2 = map(int, p["box"])
        color = POSTURE_COLOR[p["posture"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(2 * u)))
        if not crowded:
            label(img, f"#{p['id']} {POSTURE_TEXT[p['posture']]}", (x1, max(16, y1)), color, 0.5 * u)
    many_seats = status["visible"] > 25
    for seat_id, box in registry.boxes.items():
        row = status["seats"][seat_id]
        if not row["visible"]:
            continue
        occ = row["occupied"]
        color = GRAY if occ is None else (RED if occ else GREEN)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int((3 if row["kind"] == "chair" else 2) * u)))
        name = seat_id + ("*" if row["kind"] == "person" else "")
        if not many_seats:
            name += " " + ("?" if occ is None else ("Occupied" if occ else "Empty"))
            if occ and row["person_index"] is not None:
                name += f" #{sitters[row['person_index']]['id']}"
        label(img, name, (x1, max(18, y1)), color, 0.5 * u, max(1, int(2 * u)))
    hud = render_hud(status, counts, ms, u)
    m = int(12 * u)
    hh, hw = min(hud.shape[0], height - m), min(hud.shape[1], width - m)
    img[m:m + hh, m:m + hw] = hud[:hh, :hw]
    if recording:
        cv2.circle(img, (width - int(70 * u), int(28 * u)), int(9 * u), RED, -1)
        cv2.putText(img, "REC", (width - int(55 * u), int(35 * u)), FONT, 0.65 * u, RED, 2, cv2.LINE_AA)
    return img


# ---------- Ghi video, log, phiên xử lý ----------

def fmt(value):
    return "unknown" if value is None else int(value)


def to_h264(src, dst):
    """mp4v không phát được trên trình duyệt: chuyển sang H.264, trả về file dùng được."""
    done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]).returncode == 0
    if done:
        os.remove(src)
    return dst if done else src


class Recorder:
    """Ghi khung đã vẽ thành MP4 theo thời gian thật và ghi 2 log CSV.

    Khung webcam đến không đều nên khung trước được lặp lại cho đủ REC_FPS.
    *_log.csv theo định dạng vai C (t,frame,seat_id,raw_occ,smooth_occ; chỉ dòng đã biết trạng thái).
    *_full_log.csv giữ cả unknown, ghế ngoài khung, ID người và cấu hình ngưỡng, theo docs/merge-role-b.md.
    """

    def __init__(self, stem, fps, max_gap):
        self.stem, self.fps, self.max_gap = stem, fps, max_gap
        self.raw_path = os.path.join(OUT_DIR, stem + "_raw.mp4")
        self.writer, self.size, self.t0, self.last, self.written = None, None, None, None, 0
        self.flips, self.prev = {"raw": 0, "smooth": 0}, {}
        self.c_file = open(os.path.join(OUT_DIR, stem + "_log.csv"), "w", newline="")
        self.full_file = open(os.path.join(OUT_DIR, stem + "_full_log.csv"), "w", newline="")
        self.c_log, self.full_log = csv.writer(self.c_file), csv.writer(self.full_file)
        self.c_log.writerow(["t", "frame", "seat_id", "raw_occ", "smooth_occ"])
        self.full_log.writerow(["t", "frame", "seat_id", "raw_occ", "smooth_occ", "person_id", "visible", "seat_source",
                                "model", "imgsz", "conf", "overlap", "on_s", "off_s", "max_gap_s"])

    def add(self, img, t, frame_index, status):
        if self.writer is None:
            self.size = (img.shape[1], img.shape[0])
            self.writer = cv2.VideoWriter(self.raw_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            self.t0 = t
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        target = round((t - self.t0) * self.fps)
        while self.last is not None and self.written < target:
            self.writer.write(self.last)
            self.written += 1
        self.last = img
        for seat_id, row in status["seats"].items():
            raw, smooth = row["raw_occupied"], row["occupied"]
            person = "" if row.get("person_id") is None else row["person_id"]
            self.full_log.writerow([f"{t:.3f}", frame_index, seat_id, fmt(raw), fmt(smooth), person,
                                    int(row["visible"]), row["kind"], MODEL, IMGSZ, CONF, OVERLAP, ON_S, OFF_S,
                                    self.max_gap])
            if raw is not None and smooth is not None:
                self.c_log.writerow([f"{t:.3f}", frame_index, seat_id, int(raw), int(smooth)])
                last = self.prev.get(seat_id)
                if last:
                    self.flips["raw"] += last[0] != raw
                    self.flips["smooth"] += last[1] != smooth
                self.prev[seat_id] = (raw, smooth)

    def close(self):
        self.c_file.close()
        self.full_file.close()
        files = [self.stem + "_log.csv", self.stem + "_full_log.csv"]
        if self.writer is None:
            return files
        self.writer.write(self.last)
        self.writer.release()
        video = to_h264(self.raw_path, os.path.join(OUT_DIR, self.stem + ".mp4"))
        return [os.path.basename(video)] + files


class Session:
    """Một luồng hình: tracker người, bù chuyển động camera, danh sách ghế ID cố định, trạng thái ghế, bộ ghi."""

    def __init__(self, max_gap, fps, live=False):
        self.max_gap, self.fps, self.live = max_gap, fps, live  # live: webcam, hiện dấu REC khi đang quay
        self.recorder, self.frame_index, self.last_t, self.ms = None, 0, None, []
        self.reset()

    def reset(self):
        self.registry, self.engine = SeatRegistry(), TrackedEngine(self.max_gap)
        self.tracker, self.motion = make_tracker(self.fps), CameraMotion()
        self.hits, self.votes, self.ids_seen = {}, {}, set()
        self.last_chairs, self.standing_blocked = [], 0

    def lock(self):
        """Thêm ngay các chair đang thấy (không phải chờ SEAT_NEW_SECONDS)."""
        self.registry.seed(self.last_chairs)
        return len(self.registry.boxes)

    def track(self, people, kpts, frame):
        try:
            tracks = self.tracker.update(people, frame)
        except Exception:
            if len(people):
                raise
            tracks = np.empty((0, 8))  # ultralytics cũ không nhận khung rỗng
        persons = []
        for row in np.asarray(tracks).reshape(-1, 8):
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            tid, idx = int(row[4]), int(row[7])
            self.hits[tid] = self.hits.get(tid, 0) + 1
            votes = self.votes.setdefault(tid, collections.deque(maxlen=POSTURE_WINDOW))
            pose = posture(kpts[idx]) if 0 <= idx < len(kpts) else "unknown"
            if pose != "unknown":
                votes.append(pose)
            stable = collections.Counter(votes).most_common(1)[0][0] if votes else "unknown"
            persons.append({"id": tid, "box": (x1, y1, x2, y2), "score": float(np.clip(row[5], 0, 1)),
                            "posture": stable})
        return [p for p in persons if self.hits[p["id"]] >= CONFIRM_FRAMES]

    def step(self, frame, t):
        frame = fit(frame)
        height, width = frame.shape[:2]
        if self.last_t is not None:
            t = max(t, self.last_t)  # SeatState cần timestamp không giảm
        self.last_t = t
        people, kpts, chairs, ms = detect(frame)
        self.ms.append(ms)
        self.last_chairs = chairs
        self.registry.warp(self.motion.update(frame, people.xyxy))
        if self.frame_index == 0 and not self.live:
            self.registry.seed(chairs)
        persons = self.track(people, kpts, frame)
        self.ids_seen.update(p["id"] for p in persons)
        sitters = [p for p in persons if p["posture"] != "standing"]
        self.registry.update(chairs, sitters, t)
        status = self.engine.update(self.registry, [Person(p["box"], p["score"]) for p in sitters], t, width, height)
        for row in status["seats"].values():
            row["person_id"] = sitters[row["person_index"]]["id"] if row["person_index"] is not None else None
        standing = [p for p in persons if p["posture"] == "standing"]
        for seat_id, row in status["seats"].items():  # người đứng phủ ghế nhưng không được tính
            if row["visible"] and any(overlap_ratio(tuple(self.registry.boxes[seat_id]), p["box"]) >= OVERLAP
                                      for p in standing):
                self.standing_blocked += 1
        counts = {"people": len(persons), "sitting": sum(p["posture"] == "sitting" for p in persons),
                  "standing": len(standing), "posture_unknown": sum(p["posture"] == "unknown" for p in persons),
                  "ids": len(self.ids_seen)}
        out = draw(frame, self.registry, status, persons, sitters, counts, ms, self.live and self.recorder is not None)
        if self.recorder:
            self.recorder.add(out, t, self.frame_index, status)
        self.frame_index += 1
        info = {k: status[k] for k in ("total", "visible", "available", "occupied", "unknown", "out_of_view",
                                        "chair_seats", "inferred_seats")}
        return out, {**info, **counts, "ms": round(ms, 1), "recording": self.recorder is not None}


jobs = {}


def process_video(path, stem, max_seconds=None):
    """Xử lý file video trên GPU; timestamp = frame_index / fps như docs/merge-role-b.md."""
    job = jobs[stem] = {"name": stem, "progress": 0.0, "state": "running", "files": [], "note": ""}
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if max_seconds:
            total = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)
        session = Session(GAP_VIDEO, fps)
        session.recorder = Recorder(stem, fps, GAP_VIDEO)
        i, info = 0, {}
        while not stop_event.is_set() and (not total or i < total):
            ok, frame = cap.read()
            if not ok:
                break
            _, info = session.step(frame, i / fps)
            i += 1
            job["progress"] = i / total if total else 0.0
        cap.release()
        flips = dict(session.recorder.flips)
        job["files"] = session.recorder.close()
        summary = {"video": os.path.basename(path), "frames": i, "seconds": round(i / fps, 2),
                   "seats": info.get("total", 0), "chair_seats": info.get("chair_seats", 0),
                   "inferred_seats": info.get("inferred_seats", 0), "unique_person_ids": len(session.ids_seen),
                   "raw_flips": flips["raw"], "smoothed_flips": flips["smooth"],
                   "standing_blocked_seat_frames": session.standing_blocked,
                   "avg_ms": round(float(np.mean(session.ms)), 1) if session.ms else None,
                   "last_frame": {k: info.get(k) for k in ("available", "occupied", "unknown", "out_of_view",
                                                          "people", "sitting", "standing", "posture_unknown")}}
        with open(os.path.join(OUT_DIR, stem + "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        job["files"].append(stem + "_summary.json")
        job["summary"], job["avg_ms"] = summary, summary["avg_ms"]
        job["note"] = (f"{summary['seats']} ghế ({summary['chair_seats']} chair, {summary['inferred_seats']} suy ra), "
                       f"{summary['unique_person_ids']} ID người, nhấp nháy {flips['raw']}→{flips['smooth']}, "
                       f"{summary['avg_ms']} ms/khung")
        job["state"] = "done"
    except Exception as exc:  # báo lỗi lên trang, không làm sập server
        traceback.print_exc()
        job["state"] = f"error: {exc}"
    print(f">>> job {stem}: {job['state']} {job['note']} {job['files']}", flush=True)
    return job


def caption_for(job, scene):
    s = job["summary"]
    return (f"{scene}. {s['seats']} ghế ({s['chair_seats']} từ chair, {s['inferred_seats']} suy ra từ người ngồi), "
            f"{s['unique_person_ids']} ID người, {s['standing_blocked_seat_frames']} lượt người đứng che ghế bị loại.")


# ---------- Video demo sản phẩm ----------

def wrap(draw_ctx, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw_ctx.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def render_card(title, lines, kicker="", footer=""):
    """Thẻ chữ 1280x720 nền tối; lines là danh sách (chữ, màu) hoặc chuỗi."""
    w, h = DEMO_SIZE
    img = Image.new("RGB", DEMO_SIZE, (18, 21, 26))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 12, h], fill=(11, 99, 197))
    items = [(item, (200, 205, 212)) if isinstance(item, str) else item for item in lines]
    for size in (30, 28, 26, 24, 22):  # co chữ đến khi vừa khung, chừa chỗ cho footer
        title_size, line_h = int(size * 1.8), int(size * 1.4)
        title_lines = wrap(d, title, font(title_size, True), w - 180)
        body = [(part, color) for text, color in items for part in wrap(d, text, font(size), w - 180) + [None]]
        y = 100 + (48 if kicker else 0) + len(title_lines) * int(title_size * 1.25) + 24
        y += sum(line_h if part else 12 for part, _ in body)
        if y <= h - (90 if footer else 50):
            break
    y = 100
    if kicker:
        d.text((90, y), kicker.upper(), font=font(24, True), fill=(90, 165, 255))
        y += 48
    for part in title_lines:
        d.text((90, y), part, font=font(title_size, True), fill=(240, 242, 245))
        y += int(title_size * 1.25)
    y += 24
    for part, color in body:
        if part:
            d.text((90, y), part, font=font(size), fill=color)
        y += line_h if part else 12
    if footer:
        d.text((90, h - 70), footer, font=font(22), fill=(140, 148, 158))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_caption(text, tag):
    w = DEMO_SIZE[0]
    img = Image.new("RGB", (w, CAPTION_H), (12, 14, 18))
    d = ImageDraw.Draw(img)
    d.text((32, 14), tag.upper(), font=font(20, True), fill=(90, 165, 255))
    size = 26 if len(wrap(d, text, font(26), w - 64)) <= 2 else 23
    for i, part in enumerate(wrap(d, text, font(size), w - 64)[:2]):
        d.text((32, 44 + i * int(size * 1.35)), part, font=font(size), fill=(235, 237, 240))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def write_card(writer, card, seconds):
    n, fade = int(seconds * DEMO_FPS), int(0.4 * DEMO_FPS)
    for k in range(n):
        alpha = min(1.0, (k + 1) / fade, (n - k) / fade)
        writer.write(card if alpha >= 1 else (card * alpha).astype(np.uint8))


def write_clip(writer, path, caption, tag, max_seconds):
    """Chèn clip đã chạy LibSeat, co vào khung 1280x(720-CAPTION_H), giữ tỉ lệ, lấy mẫu lại DEMO_FPS."""
    w, h = DEMO_SIZE
    bar = render_caption(caption, tag)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or DEMO_FPS
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = int(min(src_total / src_fps, max_seconds) * DEMO_FPS)
    area_h = h - CAPTION_H
    read, frame = -1, None
    for k in range(n):
        target = int(k * src_fps / DEMO_FPS)
        while read < target:
            ok, nxt = cap.read()
            if not ok:
                break
            frame, read = nxt, read + 1
        if frame is None:
            break
        scale = min(w / frame.shape[1], area_h / frame.shape[0])
        fw, fh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        canvas = np.zeros((h, w, 3), np.uint8)
        x, y = (w - fw) // 2, (area_h - fh) // 2
        canvas[y:y + fh, x:x + fw] = cv2.resize(frame, (fw, fh))
        canvas[area_h:] = bar
        writer.write(canvas)
    cap.release()


def build_demo(clips, stem="libseat_demo"):
    """clips: [(đường dẫn mp4 đã chạy LibSeat, chú thích, nhãn, số giây tối đa)]."""
    job = jobs[stem] = {"name": stem, "progress": 0.0, "state": "running", "files": [], "note": ""}
    try:
        measured = [j["avg_ms"] for j in jobs.values() if j.get("avg_ms")]
        speed = (f"Đo trong lần chạy này trên {GPU_NAME}: trung bình {np.mean(measured):.0f} ms/khung "
                 f"(YOLO11s + YOLO11s-pose, {IMGSZ} px)." if measured else f"Chạy trên {GPU_NAME}.")
        cards_before = [
            (render_card("LibSeat: biết ngay ghế nào còn trống", [
                "Camera nhìn vào khu ghế, máy tính cho biết ghế nào đang có người, ghế nào trống, theo thời gian thực.",
                "Mục tiêu: sinh viên xem trước số chỗ trống thay vì đi vòng tìm ghế mùa thi."],
                kicker="Team Triangle · Vin Perception Hackathon · Track 4"), 5),
            (render_card("Cách LibSeat hoạt động", [
                "1. YOLO11s (pretrained COCO, 960 px, ngưỡng 0,10 như vai C đo) phát hiện người và ghế; YOLO11s-pose cho khớp chân.",
                "2. BoT-SORT giữ ID người khi bị che tới 3 s; người phải xuất hiện 3 khung mới được tính.",
                "3. Ghế có ID cố định, bám theo chuyển động camera, bị che không bị quên; ghế mới phải thấy liên tục 0,5 s.",
                "4. Người đang đứng không chiếm ghế. Người ngồi phủ ≥ 30% ghế thì gán, mỗi người 1 ghế.",
                "5. Làm mượt: 0,5 s mới báo có người, 1,0 s mới báo trống; ngoài khung hoặc thiếu dữ liệu thì 'chưa rõ'.",
                ("Xanh = trống  ·  Đỏ = có người  ·  Xám = chưa rõ  ·  S5* = ghế suy ra từ người ngồi", (90, 200, 120))],
                kicker="Pipeline"), 11),
        ]
        cards_after = [
            (render_card("Bằng chứng trên tập TEST có nhãn", [
                "Camera cố định (seatanalysis, 408 ghế): đúng 84,6%, balanced accuracy 81,2%.",
                "Ảnh điện thoại trong thư viện (Beyond, 1.733 ghế): balanced accuracy 69,8%.",
                "Cải thiện chủ yếu nhờ cấu hình bộ phát hiện: 62,2% → 81,2% balanced accuracy trên seatanalysis.",
                speed,
                ("Giới hạn: số đo trên là của bộ phát hiện + luật chồng hộp, chưa đo lại phần tư thế và tracker; "
                 "chân bị che thì tư thế 'chưa rõ'; ghế băng chưa ai ngồi thì chưa đếm được.", (255, 150, 150))],
                kicker="Số đo của vai C"), 11),
            (render_card("LibSeat", [
                "Demo trực tiếp: webcam laptop → GPU Kaggle qua ngrok → màn hình số chỗ trống.",
                "Mã nguồn: github.com/naoh-pt/K4-Track4-Capstone-TeamTriangle"],
                kicker="Cảm ơn", footer="Video minh họa: Pexels #9883925, #8499766 (giấy phép Pexels)."), 5),
        ]
        total = sum(s for _, s in cards_before + cards_after) + sum(c[3] for c in clips)
        raw = os.path.join(OUT_DIR, stem + "_raw.mp4")
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), DEMO_FPS, DEMO_SIZE)
        done = 0
        for card, seconds in cards_before:
            write_card(writer, card, seconds)
            done += seconds
            job["progress"] = done / total
        for path, caption, tag, seconds in clips:
            write_clip(writer, path, caption, tag, seconds)
            done += seconds
            job["progress"] = done / total
        for card, seconds in cards_after:
            write_card(writer, card, seconds)
        writer.release()
        job["files"] = [os.path.basename(to_h264(raw, os.path.join(OUT_DIR, stem + ".mp4")))]
        job["note"] = f"{len(clips)} clip, khoảng {total:.0f} s"
        job["state"] = "done"
    except Exception as exc:
        traceback.print_exc()
        job["state"] = f"error: {exc}"
    job["progress"] = 1.0
    print(f">>> job {stem}: {job['state']} {job['note']} {job['files']}", flush=True)
    return job


demo_clips = []  # clip đã chạy LibSeat, dùng lại khi dựng demo từ bản quay webcam


def clip_entry(job, caption, tag, seconds):
    if job["state"] == "done" and job["files"][0].endswith(".mp4"):
        demo_clips.append((os.path.join(OUT_DIR, job["files"][0]), caption_for(job, caption), tag, seconds))


def startup():
    """Video test (video của nhóm, demo_backup, clip Pexels) rồi dựng video demo sản phẩm."""
    team_videos = sorted(glob.glob("/kaggle/input/**/*.mp4", recursive=True))
    for n, path in enumerate(team_videos, 1):
        if stop_event.is_set():
            return
        name = os.path.splitext(secure_filename(os.path.basename(path)))[0]
        job = process_video(path, "test_" + name)
        seconds = min(20.0, job.get("summary", {}).get("seconds", 0) or 0)
        clip_entry(job, f"Video test của nhóm {n}/{len(team_videos)}", "Video test của nhóm", seconds)
    process_video(os.path.join(REPO_DIR, "demo_backup.mp4"), "demo_backup_annotated")
    for name, url, seconds, scene in FOOTAGE:
        if stop_event.is_set():
            return
        path = os.path.join(WORK_DIR, name + ".mp4")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
                f.write(resp.read())
        except Exception as exc:
            jobs[name] = {"name": name, "progress": 1.0, "state": f"error: tải clip lỗi ({exc})", "files": [], "note": ""}
            continue
        clip_entry(process_video(path, name + "_annotated", max_seconds=seconds), scene, "Video test", seconds)
    if demo_clips and not stop_event.is_set():
        build_demo(demo_clips)


cam = Session(GAP_WEBCAM, WEBCAM_FPS, live=True)
cam_lock = threading.Lock()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.post("/frame")
def frame():
    img = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "Không đọc được ảnh", 400
    t = float(request.headers.get("X-Capture-Time") or time.time())
    with cam_lock:
        out, info = cam.step(img, t)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    resp = Response(jpg.tobytes(), mimetype="image/jpeg")
    resp.headers["X-Status"] = json.dumps(info)
    return resp


@app.post("/lock")
def lock():
    with cam_lock:
        return jsonify(seats=cam.lock())


@app.post("/reset")
def reset():
    with cam_lock:
        cam.reset()
    return jsonify(ok=True)


@app.post("/record")
def record():
    on = bool((request.get_json(silent=True) or {}).get("on"))
    with cam_lock:
        if on and cam.recorder is None:
            cam.recorder = Recorder("webcam_" + time.strftime("%H%M%S"), REC_FPS, GAP_WEBCAM)
            return jsonify(recording=True)
        files = cam.recorder.close() if cam.recorder else []
        cam.recorder = None
    return jsonify(recording=False, files=files)


@app.post("/demo")
def demo():
    """Dựng lại video demo: bản quay webcam mới nhất đặt trước các clip đã có."""
    recordings = sorted(glob.glob(os.path.join(OUT_DIR, "webcam_*.mp4")), key=os.path.getmtime)
    if not recordings:
        return "Chưa có bản quay webcam nào. Hãy quay video test trước.", 400
    cap = cv2.VideoCapture(recordings[-1])
    seconds = min(90.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or REC_FPS))
    cap.release()
    clips = [(recordings[-1], "Demo trực tiếp: webcam laptop gửi khung lên GPU Kaggle qua ngrok, nhận lại trạng thái từng ghế.",
              "Demo trực tiếp", seconds)] + demo_clips
    stem = "libseat_demo_" + time.strftime("%H%M%S")
    threading.Thread(target=build_demo, args=(clips, stem), daemon=True).start()
    return jsonify(job=stem)


@app.post("/upload")
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return "Chưa chọn file", 400
    stem = os.path.splitext(secure_filename(f.filename))[0] or "video"
    stem = f"{stem}_{time.strftime('%H%M%S')}"
    path = os.path.join(WORK_DIR, secure_filename(f.filename) or "upload.mp4")
    f.save(path)
    threading.Thread(target=process_video, args=(path, stem), daemon=True).start()
    return jsonify(job=stem)


@app.get("/status")
def status():
    files = sorted(os.listdir(OUT_DIR), key=lambda n: os.path.getmtime(os.path.join(OUT_DIR, n)), reverse=True)
    left = max(0, started + SERVE_HOURS * 3600 - time.time())
    return jsonify(gpu=GPU_NAME, model=f"{MODEL} + pose {IMGSZ}px conf {CONF}", minutes_left=round(left / 60),
                   jobs=[{k: v for k, v in j.items() if k != "summary"} for j in jobs.values()],
                   files=[n for n in files if not n.endswith("_raw.mp4")])


@app.get("/files/<path:name>")
def files(name):
    return send_from_directory(OUT_DIR, name, as_attachment=request.args.get("dl") == "1")


@app.post("/shutdown")
def shutdown():
    stop_event.set()
    return jsonify(ok=True)


def load_secrets():
    for path in glob.glob("/kaggle/input/**/libseat_secrets.json", recursive=True):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def open_tunnel(secrets):
    import ngrok
    options = {"authtoken": secrets["ngrok_authtoken"]}
    if secrets.get("ngrok_domain"):
        options["domain"] = secrets["ngrok_domain"]
    return ngrok.forward(PORT, **options)


def main():
    worker = threading.Thread(target=startup, daemon=True)
    worker.start()
    secrets = load_secrets()
    if not secrets:
        print(">>> Không thấy libseat_secrets.json: chỉ tạo video test và video demo.", flush=True)
        worker.join()
        return
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False),
                     daemon=True).start()
    listener = open_tunnel(secrets)
    print(f">>> LibSeat đang chạy tại {listener.url()} trong {SERVE_HOURS} giờ", flush=True)
    deadline = started + SERVE_HOURS * 3600
    while time.time() < deadline and not stop_event.is_set():
        time.sleep(5)
    stop_event.set()
    with cam_lock:
        if cam.recorder:
            print(">>> Lưu bản ghi đang quay:", cam.recorder.close(), flush=True)
            cam.recorder = None
    end = time.time() + 120  # job đang chạy dừng ở khung kế tiếp rồi ghi file
    while any(j["state"] == "running" for j in jobs.values()) and time.time() < end:
        time.sleep(1)
    print(">>> Dừng server. Kết quả nằm trong Output của kernel:", sorted(os.listdir(OUT_DIR)), flush=True)


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LibSeat GPU</title>
<style>
:root{--bg:#f5f6f8;--fg:#16191d;--muted:#5b6470;--card:#fff;--line:#dde1e6;--accent:#0b63c5;--red:#c62828;--green:#1e8e3e;--gray:#7a828c}
@media (prefers-color-scheme:dark){:root{--bg:#0f1114;--fg:#e7e9ec;--muted:#9aa2ac;--card:#171a1f;--line:#2a2e35;--accent:#5aa5ff;--red:#ff6b6b;--green:#4cc77a;--gray:#8b939d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:16px}h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:0 0 10px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.45;cursor:default}
button.rec{color:var(--red)}button.rec.on{background:var(--red);border-color:var(--red);color:#fff}button.danger{color:var(--red)}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px;margin:12px 0}.stat b{font-size:22px;display:block}
.stat.green b{color:var(--green)}.stat.red b{color:var(--red)}.stat.gray b{color:var(--gray)}
#view{width:100%;border-radius:8px;background:#000;min-height:120px;display:block}
video.preview{width:100%;max-width:640px;border-radius:8px;background:#000;display:block;margin-top:10px}
ol{margin:8px 0;padding-left:20px}ul{padding-left:18px}li{margin:3px 0}a{color:var(--accent)}
.err{color:var(--red);min-height:1.5em}progress{width:160px;vertical-align:middle}
</style></head><body><main>
<h1>LibSeat · GPU Kaggle</h1><p class="muted" id="meta">Đang kết nối…</p>
<section class="card">
  <div class="row">
    <button id="start" class="primary">Bật webcam</button>
    <button id="lock" disabled>Thêm ngay ghế đang thấy</button>
    <button id="reset" disabled>Xóa ghế và ID</button>
    <button id="rec" class="rec" disabled>● Quay video test</button>
  </div>
  <ol class="muted">
    <li>Bật webcam khi ghế còn trống: ghế được tự thêm sau khi thấy liên tục 0,5 s và giữ nguyên ID kể cả khi bị che.</li>
    <li>Người ngồi yên 1,5 s ở chỗ chưa có ghế sẽ tạo ghế suy ra (S5*). Người đang đứng không chiếm ghế.</li>
    <li>Bấm <b>Quay video test</b> rồi diễn: ghế trống → 1 người ngồi → người đi ngang không ngồi → che ghế. Bấm lần nữa để dừng và lưu.</li>
  </ol>
  <div class="stats">
    <div class="stat green"><b id="avail">–</b><span class="muted">ghế trống</span></div>
    <div class="stat red"><b id="occ">–</b><span class="muted">có người</span></div>
    <div class="stat gray"><b id="unk">–</b><span class="muted">chưa rõ</span></div>
    <div class="stat"><b id="out">–</b><span class="muted">ngoài khung</span></div>
    <div class="stat"><b id="ppl">–</b><span class="muted">người (ngồi/đứng)</span></div>
    <div class="stat"><b id="ids">–</b><span class="muted">ID người đã thấy</span></div>
    <div class="stat"><b id="fps">–</b><span class="muted">khung/giây xử lý</span></div>
    <div class="stat"><b id="ms">–</b><span class="muted">ms suy luận GPU</span></div>
  </div>
  <p class="err" id="err"></p>
  <img id="view" alt="Khung hình đã xử lý trên GPU"><video id="cam" playsinline muted hidden></video>
</section>
<section class="card">
  <h2>Video demo sản phẩm</h2>
  <p class="muted">Tự dựng khi server khởi động từ video test. Sau khi quay bằng webcam, bấm nút dưới để dựng lại với bản quay của nhóm đặt lên đầu.</p>
  <div class="row"><button id="demo">Dựng lại video demo với bản quay mới nhất</button></div>
  <video id="demoVideo" class="preview" controls hidden></video>
</section>
<section class="card">
  <h2>Xử lý file video trên GPU</h2>
  <div class="row"><input type="file" id="file" accept="video/*"><button id="upload">Tải lên và xử lý</button></div>
  <ul id="jobs" class="muted"></ul>
</section>
<section class="card">
  <h2>Kết quả (video H.264, log CSV, tóm tắt JSON)</h2>
  <ul id="files"><li class="muted">Chưa có.</li></ul>
  <button id="shutdown" class="danger">Tắt server và lưu vào Output Kaggle</button>
</section>
</main>
<script>
const H = {"ngrok-skip-browser-warning": "1"};
const $ = id => document.getElementById(id);
const canvas = document.createElement("canvas");
let running = false, recording = false, fps = 0, shownDemo = "", shownT = 0, done = 0, fpsT0 = 0;
const IN_FLIGHT = 3;  // khứ hồi qua ngrok ~1 s: gửi song song để không phải chờ từng khung

async function post(path, body) {
  const opts = {method: "POST", headers: {...H}};
  if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

$("start").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 1280}, height: {ideal: 720}}, audio: false});
    const v = $("cam"); v.srcObject = stream; await v.play();
    running = true; $("start").disabled = true;
    for (const id of ["lock", "reset", "rec"]) $(id).disabled = false;
    loop();
  } catch (e) { $("err").textContent = "Không mở được webcam: " + e.message; }
};

function loop() {
  fpsT0 = performance.now();
  for (let i = 0; i < IN_FLIGHT; i++) setTimeout(worker, i * 200);
}

async function worker() {
  const v = $("cam");
  while (running) {
    const w = 960, h = Math.round(w * v.videoHeight / v.videoWidth);
    canvas.width = w; canvas.height = h; canvas.getContext("2d").drawImage(v, 0, 0, w, h);
    const t = Date.now() / 1000;
    const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.75));
    try {
      const r = await fetch("/frame", {method: "POST", body: blob,
        headers: {...H, "Content-Type": "image/jpeg", "X-Capture-Time": t.toFixed(3)}});
      if (!r.ok) throw new Error(await r.text());
      const s = JSON.parse(r.headers.get("X-Status"));
      const blobOut = await r.blob();
      done++;
      const now = performance.now();
      if (now - fpsT0 > 2000) { fps = done * 1000 / (now - fpsT0); done = 0; fpsT0 = now; }
      if (t < shownT) continue;  // khung về trễ hơn khung đã hiện: bỏ
      shownT = t;
      const img = $("view"), old = img.src;
      img.src = URL.createObjectURL(blobOut);
      if (old.startsWith("blob:")) URL.revokeObjectURL(old);
      $("avail").textContent = s.available; $("occ").textContent = s.occupied;
      $("unk").textContent = s.unknown; $("out").textContent = s.out_of_view;
      $("ppl").textContent = `${s.people} (${s.sitting}/${s.standing})`; $("ids").textContent = s.ids;
      $("fps").textContent = fps ? fps.toFixed(1) : "…"; $("ms").textContent = s.ms.toFixed(0);
      $("err").textContent = "";
    } catch (e) { $("err").textContent = "Lỗi gửi khung: " + e.message; await new Promise(r => setTimeout(r, 1000)); }
  }
}

$("lock").onclick = async () => {
  const r = await post("/lock");
  $("err").textContent = r.seats ? "" : "Không có hộp chair nào đủ tin cậy. Thử chỉnh góc camera hoặc ánh sáng.";
};
$("reset").onclick = () => post("/reset");
$("rec").onclick = async () => {
  const r = await post("/record", {on: !recording});
  recording = r.recording;
  $("rec").classList.toggle("on", recording);
  $("rec").textContent = recording ? "■ Dừng và lưu" : "● Quay video test";
  refresh();
};
$("demo").onclick = async () => {
  try { await post("/demo"); $("err").textContent = ""; } catch (e) { $("err").textContent = e.message; }
  refresh();
};
$("upload").onclick = async () => {
  const f = $("file").files[0]; if (!f) return;
  const fd = new FormData(); fd.append("video", f);
  $("upload").disabled = true;
  try { const r = await fetch("/upload", {method: "POST", body: fd, headers: H}); if (!r.ok) throw new Error(await r.text()); }
  catch (e) { $("err").textContent = "Tải lên lỗi: " + e.message; }
  $("upload").disabled = false; refresh();
};
$("shutdown").onclick = async () => {
  if (!confirm("Tắt server? Kernel sẽ dừng và lưu outputs/ vào Output của Kaggle.")) return;
  running = false; await post("/shutdown"); $("meta").textContent = "Server đang tắt. Tải kết quả bằng: python kaggle/push_kaggle.py output";
};

async function refresh() {
  try {
    const s = await (await fetch("/status", {headers: H})).json();
    $("meta").textContent = `${s.model} · ${s.gpu} · server còn ${s.minutes_left} phút`;
    $("jobs").innerHTML = s.jobs.map(j => `<li>${j.name}: ${j.state === "running"
      ? `<progress value="${j.progress}"></progress> ${Math.round(j.progress * 100)}%` : j.state} ${j.note ? "· " + j.note : ""}</li>`).join("");
    $("files").innerHTML = s.files.length ? s.files.map(n => `<li><a href="/files/${encodeURIComponent(n)}" target="_blank">${n}</a>
      · <a href="/files/${encodeURIComponent(n)}?dl=1">tải về</a></li>`).join("") : '<li class="muted">Chưa có.</li>';
    const demo = s.files.find(n => n.startsWith("libseat_demo") && n.endsWith(".mp4"));
    if (demo && demo !== shownDemo) { shownDemo = demo; $("demoVideo").src = "/files/" + encodeURIComponent(demo); $("demoVideo").hidden = false; }
  } catch (e) {}
}
refresh(); setInterval(refresh, 3000);
</script></body></html>
"""

if __name__ == "__main__":
    main()
