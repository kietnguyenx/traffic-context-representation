"""
Module tạo luồng video và dữ liệu telemetry giả lập thực tế cao (High-Fidelity ADAS Demo Generator)
Sử dụng ảnh góc nhìn dashcam 1080p thực tế downtown để tái hiện chính xác giao diện theo mockup:
- Drivable Area phân đoạn pixel màu xanh lục bảo (SegFormer)
- Vạch kẻ làn đường phát sáng Cyan Neon (SegFormer Lane Markings)
- Hộp bao 3D/2D xe đi trước (Lead Vehicle) với trạng thái Caution (Amber) và TTC < 1.8s
- Xe làn bên trái (Cyan Box) và Người đi bộ bên phải (Pedestrian Hazard Warning)
- Đèn tín hiệu giao thông đỏ (Red Traffic Light)
"""

import os
import cv2
import numpy as np

ASSET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dashcam_demo.jpg")


def get_base_dashcam_image(target_w=960, target_h=540):
    """Nạp ảnh nền dashcam thực tế hoặc tạo ảnh nền dự phòng chất lượng cao"""
    if os.path.exists(ASSET_PATH):
        img = cv2.imread(ASSET_PATH)
        if img is not None:
            return cv2.resize(img, (target_w, target_h))
            
    # Dự phòng nếu không tìm thấy file ảnh
    img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    img[:int(target_h * 0.52), :] = [80, 50, 40]
    img[int(target_h * 0.52):, :] = [25, 45, 25]
    road_pts = np.array([
        [int(target_w * 0.30), int(target_h * 0.52)],
        [int(target_w * 0.70), int(target_h * 0.52)],
        [target_w, target_h],
        [0, target_h]
    ], np.int32)
    cv2.fillPoly(img, [road_pts], (45, 45, 45))
    return img


def get_demo_scenario_data(num_frames=90, target_w=960, target_h=540):
    """
    Tạo danh sách khung hình và dữ liệu ngữ cảnh telemetry động cho từng frame.
    Mô phỏng kịch bản xe đi trước phanh gấp:
    - Giai đoạn 1 (0-25): Đi đều, khoảng cách 18m, TTC ~ 3.6s (Safe)
    - Giai đoạn 2 (26-65): Xe trước phanh đỏ đèn hậu, khoảng cách giảm còn 12.5m, TTC giảm xuống 1.4s (Caution in Amber)
    - Giai đoạn 3 (66-90): Hệ thống duy trì khoảng cách an toàn, tình huống ổn định trở lại
    """
    base_img = get_base_dashcam_image(target_w, target_h)
    h, w = target_h, target_w
    scenario = []

    # 1. Mặt nạ đường Drivable Area (Đa giác chuẩn phối cảnh mặt đường thành phố)
    road_poly = np.array([
        [int(w * 0.00), int(h * 0.68)],
        [int(w * 0.12), int(h * 0.55)],
        [int(w * 0.42), int(h * 0.51)],
        [int(w * 0.58), int(h * 0.51)],
        [int(w * 0.88), int(h * 0.60)],
        [int(w * 1.00), int(h * 0.72)],
        [int(w * 0.94), int(h * 0.84)],
        [int(w * 0.06), int(h * 0.84)]
    ], np.int32)
    
    drivable_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(drivable_mask, [road_poly], 255)

    # 2. Vạch kẻ phân làn (Lane Mask)
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(lane_mask, (int(w * 0.44), int(h * 0.57)), (int(w * 0.33), int(h * 0.79)), 255, 6)
    cv2.line(lane_mask, (int(w * 0.56), int(h * 0.57)), (int(w * 0.67), int(h * 0.79)), 255, 6)

    for i in range(num_frames):
        frame = base_img.copy()

        # Rung lắc camera nhẹ theo nhịp di chuyển thực tế của xe
        jitter_y = int(np.sin(i * 0.5) * 1.2)
        
        # Diễn biến của xe đi trước (Lead Vehicle #1)
        if i < 25:
            progress = i / 25.0
            dist_m = 18.0 - progress * 1.5
            ttc = 3.6 - progress * 0.4
            car_w = int(w * 0.15 + progress * (w * 0.01))
            car_h = int(h * 0.20 + progress * (w * 0.01))
            status_text = "Safe"
            is_amber = False
        elif i < 65:
            # Phanh gấp
            progress = (i - 25) / 40.0
            dist_m = 16.5 - progress * 4.0
            ttc = max(1.3, 3.2 - progress * 1.8)
            car_w = int(w * 0.16 + progress * (w * 0.025))
            car_h = int(h * 0.21 + progress * (h * 0.035))
            status_text = "Caution (amber)"
            is_amber = True
            # Giả lập đèn phanh sáng đỏ rực
            cx = int(w * 0.51)
            cy = int(h * 0.64 + jitter_y)
            cv2.circle(frame, (cx - 48, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx + 48, cy), 5, (0, 0, 255), -1)
        else:
            # Phục hồi khoảng cách
            progress = (i - 65) / 25.0
            dist_m = 12.5 + progress * 2.5
            ttc = min(2.8, 1.4 + progress * 1.4)
            car_w = int(w * 0.185 - progress * (w * 0.015))
            car_h = int(h * 0.245 - progress * (h * 0.02))
            status_text = "Caution (amber)" if ttc < 2.0 else "Normal"
            is_amber = (ttc < 2.0)

        # Tọa độ Bounding Boxes
        lead_cx = int(w * 0.51)
        lead_y2 = int(h * 0.76 + jitter_y)
        lead_bbox = [lead_cx - car_w // 2, lead_y2 - car_h, lead_cx + car_w // 2, lead_y2]

        # Xe bên trái (Lane 1 / Near)
        left_bbox = [int(w * 0.02), int(h * 0.52 + jitter_y), int(w * 0.26), int(h * 0.74 + jitter_y)]

        # Người đi bộ bên phải vỉa hè/lòng đường
        ped_x_shift = int(np.sin(i * 0.15) * 4)
        ped_bbox = [int(w * 0.89 + ped_x_shift), int(h * 0.51 + jitter_y), int(w * 0.95 + ped_x_shift), int(h * 0.63 + jitter_y)]

        # Đèn giao thông trên cao
        traffic_light_bbox = [int(w * 0.49), int(h * 0.15), int(w * 0.52), int(h * 0.26)]

        detections = [
            {
                "id": 1,
                "class": "Vehicle",
                "label": "[Car #1 | Lane 2 | Near]",
                "bbox": lead_bbox,
                "is_lead": True,
                "is_amber": is_amber,
                "distance": "NEAR",
                "ttc": round(ttc, 1)
            },
            {
                "id": 2,
                "class": "Vehicle",
                "label": "[Car #1 | Lane 2 | Near]",  # Khớp với mockup screenshot
                "bbox": left_bbox,
                "is_lead": False,
                "is_amber": False,
                "distance": "NEAR",
                "ttc": None
            },
            {
                "id": 4,
                "class": "Pedestrian",
                "label": "[Pedestrian #4 | Warning]",
                "bbox": ped_bbox,
                "is_lead": False,
                "is_amber": False,
                "distance": "NEAR",
                "ttc": None
            },
            {
                "id": 3,
                "class": "Traffic Light",
                "label": "[Traffic Light #3 | RED]",
                "bbox": traffic_light_bbox,
                "is_lead": False,
                "is_amber": False,
                "distance": "MID",
                "ttc": None
            }
        ]

        speed_kmh = max(38, int(48 - (i / num_frames) * 8 + np.sin(i * 0.2) * 2))

        # Khối log terminal JSON đồng bộ tuyệt đối với UI và mockup
        terminal_json = {
            "name": "Lane2Centers",
            "Lead Vehicle": {
                "TrafficSignal": "Red",
                "attens": 30
            },
            "Telemetry Status": {
                "Distance": "NEAR",
                "PedestrianHazard": "1 on Road"
            }
        }

        # Dữ liệu chi tiết hơn để log nếu cần
        full_context = {
            "frame": i + 1,
            "ego_lane": "Lane 2 (Center)",
            "speed_kmh": speed_kmh,
            "lead_vehicle": {
                "id": 1,
                "distance": "NEAR",
                "gap_meters": round(dist_m, 1),
                "ttc_sec": round(ttc, 1),
                "status": status_text,
                "alert": "Caution in amber" if is_amber else "Safe"
            },
            "traffic_signal": "RED (Ahead)",
            "pedestrian_hazard": "1 on Road (Alert)",
            "risk_level": "WARNING" if is_amber else "NORMAL"
        }

        scenario.append({
            "frame": frame,
            "detections": detections,
            "masks": {"drivable": drivable_mask, "lane": lane_mask},
            "telemetry": full_context,
            "terminal_json": terminal_json
        })

    return scenario
