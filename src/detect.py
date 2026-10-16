"""
Module quản lý YOLO và Tracker (trụ cột Học & Thời gian).
Thực hiện nhận diện người và ghế, duy trì ID với ByteTrack.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

class SeatDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.35):
        """
        Khởi tạo YOLO model.
        COCO class IDs: person = 0, chair = 56
        """
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf = conf
        self.person_class_id = 0
        self.chair_class_id = 56

    def detect_and_track(self, frame: np.ndarray, imgsz: int = 480) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Chạy tracking trên frame hình, tối ưu hóa cho CPU (imgsz=480, device='cpu').
        Trả về: (danh sách người, danh sách ghế)
        Mỗi item: {"box": [x1, y1, x2, y2], "conf": float, "id": Optional[int]}
        """
        # Sử dụng tracker ByteTrack tích hợp sẵn, ép device='cpu' và giảm imgsz để đạt 20-30 FPS trên CPU
        results = self.model.track(
            frame,
            classes=[self.person_class_id, self.chair_class_id],
            conf=self.conf,
            imgsz=imgsz,
            device="cpu",
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        persons = []
        chairs = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else [None] * len(boxes)

            for box, conf, cls, tid in zip(boxes, confs, clss, track_ids):
                item = {
                    "box": box.tolist(),
                    "conf": float(conf),
                    "id": int(tid) if tid is not None else None
                }
                if int(cls) == self.person_class_id:
                    persons.append(item)
                elif int(cls) == self.chair_class_id:
                    chairs.append(item)

        return persons, chairs
