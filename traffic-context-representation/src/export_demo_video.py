"""
Script Xuất Video Demo Mẫu Hoàn Chỉnh (Pre-rendered ADAS Demo Video Exporter)
Phụ trách: Thành viên 3 (Context Engine, Pipeline & Web Demo)

Chức năng:
1. Chạy luồng Pipeline tổng hợp (YOLO Detection + SegFormer Masks + Context Engine + HUD Overlay).
2. Render video hoàn chỉnh chuẩn 30 FPS với đầy đủ hiệu ứng buồng lái ADAS (Hộp bao xe trước Amber Caution,
   mặt nạ vùng đường xanh ngọc lục bảo, vạch kẻ làn Neon Cyan phát sáng, thanh Telemetry trên đầu).
3. Xuất ra file .mp4 lưu sẵn vào thư mục assets/ để phục vụ trình chiếu dự phòng khi báo cáo đồ án.
4. Hỗ trợ xuất cả kịch bản giả lập chất lượng cao (Scenario) hoặc xử lý từ file video thực tế truyền vào.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn trên Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from pipeline import DrivingPerceptionPipeline
from utils.demo_generator import get_demo_scenario_data


def draw_hud_header_banner(frame, telemetry):
    """
    Vẽ thanh thông tin Telemetry ADAS trên đầu khung hình video để khi mở
    file mp4 độc lập trên Media Player / VLC vẫn xem rõ toàn bộ chỉ số.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Banner header đen mờ phía trên (chiều cao 48px)
    banner_h = 46
    header_overlay = annotated.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, banner_h), (15, 20, 30), -1)
    annotated = cv2.addWeighted(header_overlay, 0.75, annotated, 0.25, 0)
    cv2.line(annotated, (0, banner_h), (w, banner_h), (50, 70, 90), 1)

    lead_info = telemetry.get("lead_vehicle", {})
    alert_status = lead_info.get("alert", "Safe")
    is_amber = ("Caution" in alert_status or "Amber" in alert_status)

    speed = telemetry.get("speed_kmh", 42)
    gap = lead_info.get("gap_meters", 15.0)
    ttc = lead_info.get("ttc_sec", 3.0)
    ego_lane = telemetry.get("ego_lane", "Lane 2 (Center)")

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Logo / Tên hệ thống bên trái
    cv2.circle(annotated, (18, 23), 5, (0, 255, 120), -1)  # Chấm xanh Active
    cv2.putText(annotated, "ADAS PERCEPTION HUD", (32, 28), font, 0.52, (240, 240, 240), 2, cv2.LINE_AA)

    # 2. Các chỉ số Telemetry ở giữa
    telemetry_str = f"SPEED: {speed} KM/H  |  GAP: {gap:.1f}M  |  TTC: {ttc:.1f}S  |  LANE: {ego_lane}"
    cv2.putText(annotated, telemetry_str, (int(w * 0.32), 28), font, 0.44, (200, 220, 240), 1, cv2.LINE_AA)

    # 3. Huy hiệu cảnh báo bên phải
    alert_text = "STATUS: CAUTION" if is_amber else "STATUS: SAFE"
    badge_color = (0, 165, 255) if is_amber else (60, 220, 80)  # Amber / Green
    (tw, th), _ = cv2.getTextSize(alert_text, font, 0.48, 2)
    bx1 = w - tw - 28
    cv2.rectangle(annotated, (bx1, 10), (w - 14, 36), badge_color, -1)
    cv2.putText(annotated, alert_text, (bx1 + 8, 28), font, 0.46, (15, 15, 20), 2, cv2.LINE_AA)

    return annotated


def export_scenario_demo_video(output_path, num_frames=120, fps=30, width=960, height=540):
    """
    Render video demo chất lượng cao dựa trên kịch bản tình huống thực tế
    (Xe đi trước phanh gấp, TTC < 1.8s, người đi bộ và đèn giao thông).
    """
    print(f"[1/3] Đang nạp dữ liệu kịch bản mô phỏng ({num_frames} frames)...")
    scenario_data = get_demo_scenario_data(num_frames=num_frames, target_w=width, target_h=height)

    pipeline = DrivingPerceptionPipeline(fps=fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Không thể khởi tạo VideoWriter tại: {output_path}")

    print(f"[2/3] Đang render từng khung hình với hiệu ứng ADAS HUD...")
    start_time = time.time()

    for idx, item in enumerate(scenario_data):
        frame = item["frame"]
        dets = item["detections"]
        masks = item["masks"]
        telemetry = item["telemetry"]

        # Vẽ các lớp nhận thức AI (SegFormer Emerald Green, Neon Cyan Lanes, BBox, Context Graph)
        annotated = pipeline.draw_hud_visuals(
            frame=frame,
            detections=dets,
            masks=masks,
            context_log=telemetry,
            show_masks=True,
            show_lanes=True,
            show_boxes=True,
            show_graph=True
        )

        # Chèn thêm thanh Telemetry Header trực quan lên đầu video
        final_frame = draw_hud_header_banner(annotated, telemetry)

        out.write(final_frame)
        if (idx + 1) % 30 == 0 or (idx + 1) == num_frames:
            print(f"      -> Tiến độ: {idx + 1}/{num_frames} frames ({(idx + 1) / num_frames * 100:.0f}%)")

    out.release()
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"[3/3] Hoàn tất xuất video thành công!")
    print(f"      - Đường dẫn file : {output_path}")
    print(f"      - Dung lượng     : {file_size_mb:.2f} MB")
    print(f"      - Thời lượng     : {num_frames / fps:.1f} giây ({num_frames} frames @ {fps} FPS)")
    print(f"      - Thời gian render: {elapsed:.2f} giây (~{num_frames / elapsed:.1f} FPS)")


def export_custom_video(input_video_path, output_path, fps=30):
    """
    Xử lý trực tiếp từ một file video hành trình có sẵn (.mp4, .avi).
    """
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video đầu vào: {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    pipeline = DrivingPerceptionPipeline(fps=int(src_fps))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))

    print(f"[*] Đang xử lý video đầu vào: {input_video_path} ({total_frames} frames, {width}x{height})...")
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, context_log = pipeline.run_inference(frame, frame_idx=frame_idx)
        
        # Tạo mock telemetry nếu chưa có
        telemetry = {
            "speed_kmh": 40,
            "lead_vehicle": {
                "gap_meters": 16.0,
                "ttc_sec": 2.2,
                "alert": "Caution in amber" if context_log["summary"]["risk_level"] != "An toan" else "Safe"
            },
            "ego_lane": "Lane 2 (Center)"
        }
        final_frame = draw_hud_header_banner(annotated, telemetry)
        out.write(final_frame)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"      -> Tiến độ: {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"[OK] Đã xuất video hoàn chỉnh tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Script xuất video demo ADAS Cockpit cho Thành viên 3")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn tới file video đầu vào (nếu có)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file video đầu ra (.mp4)")
    parser.add_argument("--frames", type=int, default=120, help="Số khung hình muốn render (mặc định: 120)")
    parser.add_argument("--fps", type=int, default=30, help="Số khung hình/giây (mặc định: 30)")
    args = parser.parse_args()

    # Đường dẫn xuất mặc định lưu vào src/assets/
    assets_dir = current_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = assets_dir / "demo_adas_cockpit.mp4"

    print("=" * 70)
    print("   XUẤT VIDEO KẾT QUẢ DEMO MẪU — ADAS CONTEXT PERCEPTION")
    print("   ĐỒ ÁN CƠ SỞ NGÀNH KHMT — THÀNH VIÊN 3")
    print("=" * 70)

    if args.input and os.path.exists(args.input):
        export_custom_video(args.input, out_path, fps=args.fps)
    else:
        export_scenario_demo_video(out_path, num_frames=args.frames, fps=args.fps)

    print("\n>>> SẢN PHẨM BÀN GIAO ĐÃ SẴN SÀNG ĐỂ BÁO CÁO! <<<")


if __name__ == "__main__":
    main()
