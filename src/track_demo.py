"""
ByteTrack Demo Script for Member 1
Nhiệm vụ:
- Kiểm thử giải thuật tracking ByteTrack trên video lái xe thực tế.
- Giữ ID đối tượng qua từng khung hình (Vehicle, Pedestrian, Traffic Light).
- Trích xuất điểm tiếp xúc mặt đường (bottom_center) để tích hợp với module của TV2 & TV3.
"""

import os
import sys
import cv2
import json
import argparse
from tqdm import tqdm
from ultralytics import YOLO

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm ByteTrack trên Video mẫu")
    parser.add_argument("--weights", type=str, default="src/weights/yolov8_best.pt", help="Trọng số YOLO")
    parser.add_argument("--source", type=str, default="src/assets/demo_adas_cockpit.mp4", help="Video đầu vào")
    parser.add_argument("--output", type=str, default="src/assets/demo_tracked_bytetrack.mp4", help="Video kết quả")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")

    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[!] Không tìm thấy trọng số: {args.weights}")
        return
    if not os.path.exists(args.source):
        print(f"[!] Không tìm thấy video nguồn: {args.source}")
        return

    print("=" * 65)
    print(f"🎥 CHẠY TRACKING BYTETRACK (THÀNH VIÊN 1)")
    print(f"🔹 Model:  {args.weights}")
    print(f"🔹 Source: {args.source}")
    print(f"🔹 Output: {args.output}")
    print("=" * 65)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    color_map = {
        0: (0, 255, 0),    # Xanh lá: Vehicle
        1: (255, 100, 0),  # Xanh dương: Pedestrian
        2: (0, 255, 255)   # Vàng: Traffic Light
    }

    pbar = tqdm(total=total_frames, desc="Processing Video Tracking")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=args.conf, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                coords = boxes.xyxy.cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()

                for tid, (x1, y1, x2, y2), cid, conf in zip(track_ids, coords, clss, confs):
                    cname = model.names.get(cid, str(cid))
                    color = color_map.get(cid, (255, 255, 255))

                    # Vẽ BBox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Điểm tiếp xúc mặt đường (Bottom Center)
                    bottom_x = int((x1 + x2) / 2.0)
                    bottom_y = int(y2)
                    cv2.circle(frame, (bottom_x, bottom_y), 4, (0, 0, 255), -1)

                    # Nhãn ID và Tên lớp
                    label = f"ID:{tid} {cname} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"\n✅ Đã xuất video tracking thành công tại: {args.output}")

if __name__ == "__main__":
    main()
