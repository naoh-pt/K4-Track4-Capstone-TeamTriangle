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
