import numpy as np
import cv2

class TrafficContextEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.prev_boxes = {}  # Lưu track_id: (area, frame_idx)

    def estimate_distance(self, bbox_bottom_y, frame_height):
        """Ước lượng khoảng cách tương đối (Gần / Vừa / Xa) dựa trên vị trí đáy hộp bao"""
        rel_y = bbox_bottom_y / frame_height
        if rel_y > 0.75:
            return "Gan"
        elif rel_y > 0.50:
            return "Vua"
        else:
            return "Xa"

    def estimate_ttc(self, track_id, current_area, current_frame):
        """Tính toán thời gian tới va chạm (TTC - Time to Collision)"""
        ttc = float('inf')
        if track_id in self.prev_boxes:
            prev_area, prev_frame = self.prev_boxes[track_id]
            delta_frames = current_frame - prev_frame
            if delta_frames > 0 and current_area > prev_area:
                area_rate = (current_area - prev_area) / delta_frames
                ttc = (current_area / area_rate) / self.fps
        self.prev_boxes[track_id] = (current_area, current_frame)
        return ttc

    def check_in_drivable(self, point, drivable_mask):
        """Kiểm tra chân hộp bao có nằm trong vùng di chuyển cho phép (Drivable Area) không"""
        if drivable_mask is None or drivable_mask.size == 0:
            return True
        x, y = int(point[0]), int(point[1])
        h, w = drivable_mask.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # Lấy vùng lân cận 3x3 để khử nhiễu
            patch = drivable_mask[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
            return np.any(patch > 0)
        return False

    def process_frame(self, detections, masks, weather_label="Nang rao", frame_idx=0, frame_shape=(384, 640)):
        """
        Xử lý suy luận ngữ cảnh và xuất ra cấu trúc dữ liệu JSON
        - detections: danh sách dict [{'id': int, 'class': str, 'bbox': [x1, y1, x2, y2]}, ...]
        - masks: dict {'drivable': np.ndarray, 'lane': np.ndarray}
        """
        h, w = frame_shape
        context_data = {
            "frame": frame_idx,
            "weather": weather_label,
            "warnings": [],
            "objects": []
        }

        # Siết ngưỡng an toàn TTC khi gặp thời tiết xấu (mưa, ban đêm)
        ttc_threshold = 2.5 if weather_label in ["Mua", "Ban dem"] else 1.8
        drivable_mask = masks.get('drivable', None) if masks else None

        for obj in detections:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            bottom_center = ((x1 + x2) / 2, y2)
            area = max(1, (x2 - x1) * (y2 - y1))
            
            distance = self.estimate_distance(y2, h)
            is_in_road = self.check_in_drivable(bottom_center, drivable_mask)
            ttc = self.estimate_ttc(obj['id'], area, frame_idx)

            risk_level = "Thap"
            if is_in_road and distance == "Gan":
                risk_level = "Cao"
            elif is_in_road and (distance == "Vua" or ttc < ttc_threshold):
                risk_level = "Trung binh"

            # Tổng hợp cảnh báo quan trọng
            if obj['class'] == 'Pedestrian' and is_in_road:
                context_data["warnings"].append(f"NGUY HIEM: Nguoi di bo tren duong (ID: {obj['id']})")
            elif obj['class'] == 'Vehicle' and risk_level == "Cao":
                ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "Khong xac dinh"
                context_data["warnings"].append(f"CANH BAO: Xe truoc qua gan (ID: {obj['id']}, TTC: {ttc:.1f}s)")

            context_data["objects"].append({
                "id": obj['id'],
                "class": obj['class'],
                "distance": distance,
                "in_drivable": bool(is_in_road),
                "ttc": round(ttc, 2) if ttc != float('inf') else None,
                "risk_level": risk_level
            })

        return context_data