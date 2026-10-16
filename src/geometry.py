"""Image overlap and calibrated floor geometry (no detector dependency)."""
import json
import math
from dataclasses import dataclass
from pathlib import Path


def validate_box(box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("Box must contain four finite coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box must have positive width and height")
    return tuple(box)


def foot_point(box):
    x1, _, x2, y2 = validate_box(box)
    return ((x1 + x2) / 2, y2)


def overlap_ratio(seat, person):
    sx1, sy1, sx2, sy2 = validate_box(seat)
    px1, py1, px2, py2 = validate_box(person)
    intersection = max(0, min(sx2, px2) - max(sx1, px1)) * max(
        0, min(sy2, py2) - max(sy1, py1))
    return intersection / ((sx2 - sx1) * (sy2 - sy1))


class FloorCalibration:
    """Four ordered convex image points mapped to a floor rectangle in metres."""

    def __init__(self, image_points, width_m, height_m):
        import cv2
        import numpy as np

        points = np.asarray(image_points, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Expected four finite image points")
        if not all(math.isfinite(v) and v > 0 for v in (width_m, height_m)):
            raise ValueError("Floor dimensions must be positive and finite")
        if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 1:
            raise ValueError("Points must form an ordered, non-degenerate convex quadrilateral")
        target = np.float32([[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]])
        self.matrix = cv2.getPerspectiveTransform(points, target)
        if not np.isfinite(self.matrix).all() or np.linalg.matrix_rank(self.matrix) < 3:
            raise ValueError("Invalid homography")
        self.image_points = points.tolist()
        self.width_m, self.height_m = width_m, height_m

    def to_floor(self, point):
        if len(point) != 2 or not all(math.isfinite(v) for v in point):
            raise ValueError("Expected a finite 2D point")
        x, y = point
        h = self.matrix
        denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        if abs(denominator) < 1e-9:
            raise ValueError("Point lies on the projective horizon")
        return ((h[0, 0] * x + h[0, 1] * y + h[0, 2]) / denominator,
                (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / denominator)

    def save(self, path):
        Path(path).write_text(json.dumps({"image_points": self.image_points,
            "width_m": self.width_m, "height_m": self.height_m}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Person:
    box: tuple
    confidence: float = 1.0
    floor_anchor: tuple | None = None  # image coordinates of the floor contact point


@dataclass(frozen=True)
class Seat:
    seat_id: str
    box: tuple
    floor_anchor: tuple | None = None


def assign_people(seats, people, calibration=None, overlap_threshold=0.3,
                  max_distance_m=0.6, min_confidence=0.25):
    """Return seat_id -> person index or None; greedy one-to-one matching.

    Prefer largest seat overlap, then shortest floor distance. This heuristic
    is intentionally simple; it does not infer whether a person is sitting.
    """
    if not 0 < overlap_threshold <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("Invalid overlap/confidence threshold")
    if not math.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("Distance threshold must be positive and finite")
    seats, people = list(seats), list(people)
    if len({s.seat_id for s in seats}) != len(seats):
        raise ValueError("Seat IDs must be unique")
    for seat in seats:
        validate_box(seat.box)
    for person in people:
        validate_box(person.box)
        if not math.isfinite(person.confidence) or not 0 <= person.confidence <= 1:
            raise ValueError("Confidence must be between zero and one")
    candidates = []
    for si, seat in enumerate(seats):
        for pi, person in enumerate(people):
            if person.confidence < min_confidence:
                continue
            overlap = overlap_ratio(seat.box, person.box)
            if overlap < overlap_threshold:
                continue
            distance = 0.0
            if calibration is not None:
                a = calibration.to_floor(seat.floor_anchor if seat.floor_anchor is not None else foot_point(seat.box))
                b = calibration.to_floor(person.floor_anchor if person.floor_anchor is not None else foot_point(person.box))
                distance = math.dist(a, b)
                if distance > max_distance_m:
                    continue
            candidates.append((-overlap, distance, si, pi))
    matches = {s.seat_id: None for s in seats}
    used_people = set()
    for _, _, si, pi in sorted(candidates):
        key = seats[si].seat_id
        if matches[key] is None and pi not in used_people:
            matches[key] = pi
            used_people.add(pi)
    return matches


# --- API của vai A (nhánh phat), dùng bởi src/app.py ---
# foot_point ở trên tương đương bản của nhánh phat, có thêm kiểm tra hộp hợp lệ.

def compute_iou(box1, box2):
    """Tính Intersection-over-Union giữa 2 bounding box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    return inter_area / (area1 + area2 - inter_area)


def center_point(box):
    """Lấy tâm của bounding box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class FloorHomography:
    """Xử lý phép biến đổi phối cảnh từ ảnh sang mặt phẳng sàn (mét)."""
    def __init__(self, img_pts=None, floor_pts=None):
        """
        img_pts: 4 điểm trên ảnh [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        floor_pts: 4 điểm thực trên sàn (mét) [[0, 0], [W, 0], [W, H], [0, H]]
        """
        self.H = None
        if img_pts is not None and floor_pts is not None:
            self.set_points(img_pts, floor_pts)

    def set_points(self, img_pts, floor_pts):
        import cv2
        import numpy as np

        self.H = cv2.getPerspectiveTransform(img_pts.astype(np.float32), floor_pts.astype(np.float32))

    def to_floor(self, pt):
        import cv2
        import numpy as np

        if self.H is None:
            return None
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        res = cv2.perspectiveTransform(p, self.H)
        return float(res[0, 0, 0]), float(res[0, 0, 1])

    def floor_distance(self, pt1, pt2):
        """Tính khoảng cách euclid trên mặt sàn (mét)."""
        f1 = self.to_floor(pt1)
        f2 = self.to_floor(pt2)
        if f1 is None or f2 is None:
            return None
        return math.hypot(f1[0] - f2[0], f1[1] - f2[1])
