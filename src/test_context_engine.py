"""
Script kiểm thử độc lập cho TrafficContextEngine
Chạy kiểm tra logic suy luận ngữ cảnh không cần nạp trọng số mô hình nặng.
"""

import json
import numpy as np
import sys
from pathlib import Path

# Đảm bảo in tiếng Việt không bị lỗi encoding trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Đảm bảo import được context_engine trong thư mục src
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from context_engine import TrafficContextEngine


def create_synthetic_masks(h=384, w=640):
    """
    Tạo mặt nạ giả lập:
    - drivable_mask: Vùng đường hình thang từ giữa ảnh xuống đáy
    - lane_mask: 2 vạch kẻ phân làn trắng dọc theo đường
    """
    drivable_mask = np.zeros((h, w), dtype=np.uint8)
    # Lòng đường: nửa dưới màn hình
    pts = np.array([
        [int(w * 0.25), int(h * 0.45)],
        [int(w * 0.75), int(h * 0.45)],
        [w, h],
        [0, h]
    ], np.int32)
    cv2_polyfill = False
    try:
        import cv2
        cv2.fillPoly(drivable_mask, [pts], 255)
        cv2_polyfill = True
    except ImportError:
        # Fallback nếu chưa cài opencv
        drivable_mask[int(h * 0.5):, :] = 255

    lane_mask = np.zeros((h, w), dtype=np.uint8)
    # Vẽ 2 vạch làn dọc
    lane_mask[int(h * 0.45):, int(w * 0.38)] = 255
    lane_mask[int(h * 0.45):, int(w * 0.62)] = 255

    return {"drivable": drivable_mask, "lane": lane_mask}


def run_unit_test():
    print("=" * 60)
    print("  KIỂM THỬ ĐỘC LẬP: TRAFFIC CONTEXT ENGINE (THÀNH VIÊN 3)")
    print("=" * 60)

    engine = TrafficContextEngine(fps=30)
    h, w = 384, 640
    masks = create_synthetic_masks(h, w)

    # --- FRAME 0: Xe ở khoảng cách vừa, có người đi bộ và đèn giao thông ---
    print("\n--- [1] Xử lý Frame 0 (Khởi tạo theo dõi) ---")
    detections_f0 = [
        # Xe #1 ở cự ly vừa
        {"id": 1, "class": "Vehicle", "bbox": [280, 200, 360, 270]},
        # Người đi bộ #2 đang trên đường
        {"id": 2, "class": "Pedestrian", "bbox": [220, 240, 250, 310]},
        # Đèn giao thông #3 ở xa phía trên
        {"id": 3, "class": "Traffic Light", "bbox": [310, 60, 335, 110]}
    ]

    res_f0 = engine.process_frame(
        detections=detections_f0,
        masks=masks,
        weather_label="Nang rao",
        frame_idx=0,
        frame_shape=(h, w)
    )

    print(f"Tổng số xe: {res_f0['summary']['total_vehicles']}")
    print(f"Người đi bộ trên đường: {res_f0['summary']['pedestrians_on_road']}")
    print(f"Phát hiện đèn giao thông: {res_f0['summary']['traffic_light_ahead']} (K/c: {res_f0['summary']['traffic_light_distance']})")
    print(f"Mức độ rủi ro chung: {res_f0['summary']['risk_level']}")
    print("Các cảnh báo kích hoạt:")
    for warn in res_f0["warnings"]:
        print(f"  -> {warn}")

    # --- FRAME 5: Xe #1 tiến lại rất gần (BBox phình to gấp đôi) -> TTC kích hoạt ---
    print("\n--- [2] Xử lý Frame 5 (Xe #1 lao tới gần - Kiểm tra TTC & Cảnh báo FCW) ---")
    detections_f5 = [
        # Xe #1 phình to diện tích và dịch xuống đáy ảnh
        {"id": 1, "class": "Vehicle", "bbox": [250, 220, 390, 350]},
        # Người đi bộ #2 vẫn trên đường
        {"id": 2, "class": "Pedestrian", "bbox": [215, 240, 245, 310]},
        # Đèn giao thông #3
        {"id": 3, "class": "Traffic Light", "bbox": [310, 60, 335, 110]}
    ]

    res_f5 = engine.process_frame(
        detections=detections_f5,
        masks=masks,
        weather_label="Nang rao",
        frame_idx=5,
        frame_shape=(h, w)
    )

    print(f"Mức độ rủi ro: {res_f5['summary']['risk_level']}")
    print(f"Số cảnh báo: {res_f5['summary']['active_warnings_count']}")
    for warn in res_f5["warnings"]:
        print(f"  -> {warn}")

    print("\nChi tiết đối tượng [Xe #1] sau khi tính TTC:")
    for obj in res_f5["objects"]:
        if obj["id"] == 1:
            print(f"  - Mô tả: {obj['description']}")
            print(f"  - TTC: {obj['ttc']} giây")
            print(f"  - Vị trí làn: {obj['lane_position']}")
            print(f"  - Trong đường: {obj['in_drivable']}")

    print("\n" + "=" * 60)
    print("  KẾT QUẢ JSON ĐẦU RA MẪU CHO 1 FRAME (TRÍCH ĐOẠN)")
    print("=" * 60)
    sample_output = {
        "frame": res_f5["frame"],
        "weather": res_f5["weather"],
        "summary": res_f5["summary"],
        "warnings": res_f5["warnings"],
        "objects": res_f5["objects"]
    }
    print(json.dumps(sample_output, indent=2, ensure_ascii=False))
    print("\n>>> KIỂM THỬ HOÀN TẤT THÀNH CÔNG! <<<")


if __name__ == "__main__":
    run_unit_test()
