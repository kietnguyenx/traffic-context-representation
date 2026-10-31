"""
Công cụ kiểm thử nhận diện nhanh (Quick Detection Test Tool)
Hỗ trợ kiểm tra trực tiếp ảnh hoặc video bất kỳ với mô hình yolo_best_v2.pt (100K Trained)
Tác giả: Thành viên 1 - Module Object Detection
"""

import sys
import os
import argparse
import time
import cv2

# Thêm src vào đường dẫn
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from pipeline import DrivingPerceptionPipeline

def main():
    parser = argparse.ArgumentParser(description="Kiểm thử nhận diện YOLOv8s BDD100K (100K Trained)")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join(src_dir, "assets", "dashcam_demo.jpg"),
        help="Đường dẫn tới file ảnh (.jpg, .png) hoặc video (.mp4)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.18,
        help="Ngưỡng tin cậy (Mặc định: 0.18 để bắt trọn đèn giao thông ở xa)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join(cur_dir, "ket_qua_test.jpg"),
        help="Đường dẫn file kết quả lưu ra"
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[LỖI] Không tìm thấy file đầu vào: {input_path}")
        return

    print("=" * 65)
    print("HỆ THỐNG KIỂM THỬ NHẬN DIỆN NGỮ CẢNH GIAO THÔNG (BDD100K 100K)")
    print(f"File kiểm thử: {input_path}")
    print(f"Ngưỡng Conf  : {args.conf}")
    print("=" * 65)

    pipeline = DrivingPerceptionPipeline()

    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

    if is_image:
        frame = cv2.imread(input_path)
        if frame is None:
            print("[LỖI] Không thể đọc được file ảnh.")
            return

        annotated, log = pipeline.run_inference(frame, frame_idx=0, conf=args.conf)
        summary = log.get("summary", {})
        objects = log.get("objects", [])

        tls = [o for o in objects if o['class'] == 'Traffic Light']
        vehs = [o for o in objects if o['class'] == 'Vehicle']
        peds = [o for o in objects if o['class'] == 'Pedestrian']

        print(f"\n[KẾT QUẢ PHÂN TÍCH]")
        print(f" -> Đèn giao thông (Traffic Light): {len(tls)} vật thể")
        print(f" -> Xe cộ (Vehicle)                : {len(vehs)} phương tiện")
        print(f" -> Người đi bộ (Pedestrian)       : {len(peds)} người")
        print(f" -> Cảnh báo đèn phía trước        : {summary.get('traffic_light_ahead', False)}")
        print(f" -> Mức độ rủi ro va chạm          : {summary.get('risk_level', 'An toan')}")

        out_path = args.output
        if not out_path.lower().endswith(('.jpg', '.png')):
            out_path = os.path.splitext(out_path)[0] + ".jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\n[XUẤT THÀNH PHẨM] Đã lưu ảnh kết quả trực quan hóa tại: {out_path}")

    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("[LỖI] Không thể mở video.")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = args.output
        if not out_path.lower().endswith(('.mp4', '.avi')):
            out_path = os.path.splitext(out_path)[0] + ".mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx = 0
        total_tl_detections = 0
        print(f"Đang xử lý {total_f} frames video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, log = pipeline.run_inference(frame, frame_idx=frame_idx, conf=args.conf)
            writer.write(annotated)
            tls = [o for o in log.get("objects", []) if o['class'] == 'Traffic Light']
            total_tl_detections += len(tls)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f" -> Đã phân tích {frame_idx}/{total_f} frames...")

        cap.release()
        writer.release()
        print(f"\n[HOÀN TẤT] Video đã được lưu tại: {out_path}")
        print(f"Tổng số lượt phát hiện đèn giao thông qua các frame: {total_tl_detections}")

    print("=" * 65)

if __name__ == "__main__":
    main()
