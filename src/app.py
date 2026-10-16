"""
Ứng dụng chính LibSeat (vai A: Pipeline, hiển thị, đo độ trễ & FPS).
Hỗ trợ chạy từ Webcam (0), file video, hoặc video test.
Có phím tắt 'c' để cấu hình ghế cố định hoặc lấy ghế động từ YOLO.
Ghi log CSV phục vụ vai C đo đạc bằng chứng.
"""

import cv2
import time
import csv
import os
import argparse
import numpy as np

from detect import SeatDetector
from geometry import compute_iou, foot_point, center_point, FloorHomography
from smoothing import FrameSeatState as SeatState

def parse_args():
    parser = argparse.ArgumentParser(description="LibSeat - Library Seat Occupancy Detection")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) hoặc đường dẫn video file")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO checkpoint")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou-thr", type=float, default=0.20, help="Ngưỡng IoU người-ghế để tính có người")
    parser.add_argument("--dist-thr", type=float, default=0.75, help="Ngưỡng khoảng cách sàn (mét) nếu dùng homography")
    parser.add_argument("--log", type=str, default="data/logs/run_log.csv", help="Đường dẫn lưu log CSV")
    return parser.parse_args()

def main():
    args = parse_args()

    # Nhận diện nguồn video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Lỗi] Không thể mở nguồn video: {args.source}")
        return

    print(">>> Khởi tạo LibSeat SeatDetector...")
    detector = SeatDetector(model_name=args.model, conf=args.conf)
    
    # Chuẩn bị file log
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log_file = open(args.log, mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "fps", "total_seats", "empty_seats", "occupied_seats", "raw_states", "smooth_states"])

    # Khởi tạo danh sách ghế cố định (nếu muốn định nghĩa ghế cho góc quay cố định)
    # Người dùng có thể nhấn 's' để chốt các vị trí ghế hiện tại
    fixed_seats = None
    seat_states = {}
    
    # Quản lý FPS
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    print(">>> LibSeat đang chạy. Nhấn 'q' để thoát, 'r' để reset, 's' để chốt vị trí ghế cố định.")

    is_video_file = not str(args.source).isdigit()

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[Thông báo] Hết luồng video / webcam ngắt kết nối.")
                break

        current_time = time.time()
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / (current_time - fps_start_time)
            fps_start_time = current_time

        # 1. Pipeline Detect & Track
        persons, detected_chairs = detector.detect_and_track(frame)

        # Nếu chưa chốt ghế cố định, dùng các ghế detect được từ YOLO
        active_chairs = fixed_seats if fixed_seats is not None else detected_chairs

        total_seats = len(active_chairs)
        occupied_count = 0
        raw_states = []
        smooth_states = []

        for idx, chair in enumerate(active_chairs):
            cbox = chair["box"]
            seat_id = chair.get("id") if chair.get("id") is not None else idx

            # Kiểm tra xem có người nào trùng vào ghế không
            is_occupied_raw = False
            for person in persons:
                pbox = person["box"]
                iou = compute_iou(cbox, pbox)
                if iou >= args.iou_thr:
                    is_occupied_raw = True
                    break

            raw_states.append(1 if is_occupied_raw else 0)

            # Cập nhật Hysteresis (Vai B)
            if seat_id not in seat_states:
                seat_states[seat_id] = SeatState(seat_id=seat_id, on_frames=10, off_frames=25)

            is_occupied_smooth = seat_states[seat_id].update(is_occupied_raw)
            smooth_states.append(1 if is_occupied_smooth else 0)

            if is_occupied_smooth:
                occupied_count += 1

            # 3. Vẽ lên màn hình: Đỏ nếu có người, Xanh lá nếu trống
            color = (0, 0, 255) if is_occupied_smooth else (0, 255, 0)
            status_text = "Occupied" if is_occupied_smooth else "Empty"
            x1, y1, x2, y2 = map(int, cbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Seat {seat_id}: {status_text}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Vẽ bounding box người (màu xám / vàng nhạt)
        for person in persons:
            px1, py1, px2, py2 = map(int, person["box"])
            pid = person["id"]
            label = f"Person #{pid}" if pid is not None else "Person"
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 200, 0), 1)
            cv2.putText(frame, label, (px1, max(15, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        empty_seats = max(0, total_seats - occupied_count)

        # 4. Hiển thị Dashboard trên frame
        # Hộp thông tin trên cùng
        cv2.rectangle(frame, (10, 10), (380, 85), (0, 0, 0), -1)
        header_text = f"Con {empty_seats}/{total_seats} cho trong"
        cv2.putText(frame, header_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        info_text = f"FPS: {fps:.1f} | Nguoi: {len(persons)} | Ghe: {total_seats}"
        cv2.putText(frame, info_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Ghi CSV Log
        csv_writer.writerow([f"{current_time:.3f}", f"{fps:.1f}", total_seats, empty_seats, occupied_count, raw_states, smooth_states])

        # Hiển thị cửa sổ
        cv2.imshow("LibSeat - Live Occupancy Map", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            seat_states.clear()
            fixed_seats = None
            print(">>> Reset seat states và bỏ chốt ghế cố định.")
        elif key == ord('s'):
            if detected_chairs:
                fixed_seats = detected_chairs.copy()
                print(f">>> Đã chốt {len(fixed_seats)} vị trí ghế cố định cho phòng học/thư viện!")
            else:
                print(">>> Chưa nhận diện được ghế nào để chốt.")

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print(f">>> Đã dừng. Log lưu tại {args.log}")

if __name__ == "__main__":
    main()
