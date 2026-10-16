"""Usage: python -m src.calibrate image.jpg --width 2 --height 1.5"""
import argparse
from pathlib import Path
from .geometry import FloorCalibration


def main():
    import cv2

    parser = argparse.ArgumentParser(description="Click floor corners: TL, TR, BR, BL")
    parser.add_argument("image")
    parser.add_argument("--width", type=float, required=True, help="Floor width in metres")
    parser.add_argument("--height", type=float, required=True, help="Floor height in metres")
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output exists; choose a new --output path")
    frame = cv2.imread(args.image)
    if frame is None:
        parser.error("Cannot read image")
    points = []
    window = "Floor calibration: click TL TR BR BL; R reset; Enter save; Esc cancel"

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            display = frame.copy()
            for i, point in enumerate(points):
                cv2.circle(display, point, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                try:
                    calibration = FloorCalibration(points, args.width, args.height)
                except ValueError as error:
                    print(error)
                    points.clear()
                    continue
                calibration.save(args.output)
                print(f"Saved {args.output}")
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
