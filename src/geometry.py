"""
Module xử lý hình học (trụ cột Hình học).
Bao gồm:
- Tính toán IoU giữa hộp người và ghế.
- Homography chuyển tọa độ chân người / tâm ghế sang mặt sàn phẳng (mét).
- Gán người vào ghế kết hợp IoU và khoảng cách thực tế trên sàn.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

def compute_iou(box1: List[float], box2: List[float]) -> float:
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

def foot_point(box: List[float]) -> Tuple[float, float]:
    """Lấy điểm chân (điểm chính giữa cạnh dưới của bounding box)."""
    return ((box[0] + box[2]) / 2.0, box[3])

def center_point(box: List[float]) -> Tuple[float, float]:
    """Lấy tâm của bounding box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

class FloorHomography:
    """Xử lý phép biến đổi phối cảnh từ ảnh sang mặt phẳng sàn (mét)."""
    def __init__(self, img_pts: Optional[np.ndarray] = None, floor_pts: Optional[np.ndarray] = None):
        """
        img_pts: 4 điểm trên ảnh [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        floor_pts: 4 điểm thực trên sàn (mét) [[0, 0], [W, 0], [W, H], [0, H]]
        """
        self.H = None
        if img_pts is not None and floor_pts is not None:
            self.set_points(img_pts, floor_pts)

    def set_points(self, img_pts: np.ndarray, floor_pts: np.ndarray):
        self.H = cv2.getPerspectiveTransform(img_pts.astype(np.float32), floor_pts.astype(np.float32))

    def to_floor(self, pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.H is None:
            return None
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        res = cv2.perspectiveTransform(p, self.H)
        return float(res[0, 0, 0]), float(res[0, 0, 1])

    def floor_distance(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Optional[float]:
        """Tính khoảng cách euclid trên mặt sàn (mét)."""
        f1 = self.to_floor(pt1)
        f2 = self.to_floor(pt2)
        if f1 is None or f2 is None:
            return None
        return float(np.hypot(f1[0] - f2[0], f1[1] - f2[1]))
