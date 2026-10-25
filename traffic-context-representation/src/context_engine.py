"""
Module Suy Luận Ngữ Cảnh Giao Thông (Traffic Context Representation Engine)
Được phát triển cho Đồ án Cơ sở Ngành KHMT - Đề tài số 2:
Nhận diện và biểu diễn ngữ cảnh các nhân tố ảnh hưởng đến hoạt động giao thông.

Tác giả: Thành viên 3 (Phụ trách Context Engine & Web Demo)
"""

import numpy as np
import cv2


class TrafficContextEngine:
    """
    Bộ suy luận ngữ cảnh giao thông kết hợp đầu ra từ:
    - Object Detection + Tracking (YOLOv8 + ByteTrack): BBoxes, Class, Track ID
    - Semantic Segmentation (SegFormer): Mask Drivable Area & Lane Markings
    """

    def __init__(self, fps=30):
        self.fps = fps
        self.prev_boxes = {}  # Lưu trữ: track_id -> (area, frame_idx)

    def estimate_distance(self, bbox, frame_shape, obj_class="Vehicle"):
        """
        Ước lượng khoảng cách tương đối (Gần / Vừa / Xa) dựa trên vị trí và kích thước BBox.
        - Với Vehicle và Pedestrian: Dựa trên tọa độ y của điểm tiếp xúc mặt đường (đáy BBox).
        - Với Traffic Light: Dựa trên chiều cao tương đối của hộp bao trong khung hình.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        if obj_class == "Traffic Light":
            rel_height = (y2 - y1) / h
            if rel_height > 0.12:
                return "Gan"
            elif rel_height > 0.06:
                return "Vua"
            else:
                return "Xa"
        else:
            rel_y = y2 / h
            if rel_y > 0.70:
                return "Gan"
            elif rel_y > 0.48:
                return "Vua"
            else:
                return "Xa"

    def estimate_ttc(self, track_id, current_area, current_frame):
        """
        Tính toán chỉ số Thời gian tới va chạm (TTC - Time To Collision)
        dựa trên tốc độ tăng diện tích của Bounding Box qua các frame:
            area_rate = (current_area - prev_area) / delta_frames
            TTC = (current_area / area_rate) / fps
        """
        ttc = float('inf')
        if track_id in self.prev_boxes:
            prev_area, prev_frame = self.prev_boxes[track_id]
            delta_frames = current_frame - prev_frame
            if delta_frames > 0 and current_area > prev_area:
                area_rate = (current_area - prev_area) / delta_frames
                if area_rate > 0:
                    ttc = (current_area / area_rate) / self.fps

        # Cập nhật thông tin frame hiện tại
        self.prev_boxes[track_id] = (current_area, current_frame)
        return ttc

    def check_in_drivable(self, point, drivable_mask):
        """
        Kiểm tra điểm tiếp xúc mặt đường (chân BBox) có nằm trong vùng di chuyển (Drivable Area) không.
        Lấy mẫu vùng cửa sổ 5x5 xung quanh điểm để tăng khả năng chịu nhiễu biên.
        """
        if drivable_mask is None or drivable_mask.size == 0:
            return True  # Mặc định an toàn nếu chưa nạp mask

        x, y = int(point[0]), int(point[1])
        h, w = drivable_mask.shape[:2]

        if 0 <= x < w and 0 <= y < h:
            y_min = max(0, y - 2)
            y_max = min(h, y + 3)
            x_min = max(0, x - 2)
            x_max = min(w, x + 3)
            patch = drivable_mask[y_min:y_max, x_min:x_max]
            return bool(np.any(patch > 0))
        return False

    def determine_lane_position(self, bottom_point, lane_mask):
        """
        Quét ngang mask Lane tại hàng ảnh y của điểm tiếp xúc mặt đường
        để đếm số lượng vạch kẻ và xác định vị trí làn xe tương đối (Làn trái / Làn giữa / Làn phải).
        """
        if lane_mask is None or lane_mask.size == 0:
            return "Khong ro lan"

        x, y = int(bottom_point[0]), int(bottom_point[1])
        h, w = lane_mask.shape[:2]

        if not (0 <= y < h and 0 <= x < w):
            return "Ngoai khung hinh"

        scan_line = lane_mask[y, :]
        lane_indices = np.where(scan_line > 0)[0]

        if len(lane_indices) == 0:
            return "Cung lan / Dang di chuyen"

        # Gom cụm các điểm vạch kẻ gần nhau (<= 15px) thành từng vạch phân làn riêng biệt
        lanes_x = []
        cluster = [lane_indices[0]]
        for idx in lane_indices[1:]:
            if idx - cluster[-1] <= 15:
                cluster.append(idx)
            else:
                lanes_x.append(int(np.mean(cluster)))
                cluster = [idx]
        lanes_x.append(int(np.mean(cluster)))

        lanes_left = [lx for lx in lanes_x if lx < x]
        lanes_right = [lx for lx in lanes_x if lx > x]

        if not lanes_left and lanes_right:
            return "Lan ngoai cung ben trai"
        elif lanes_left and not lanes_right:
            return "Lan ngoai cung ben phai"
        elif lanes_left and lanes_right:
            return f"Lan giua (giua vach {len(lanes_left)} va {len(lanes_left) + 1})"

        return "Cung lan"

    def process_frame(self, detections, masks=None, weather_label="Nang rao", frame_idx=0, frame_shape=(384, 640)):
        """
        Xử lý toàn bộ đối tượng và mặt nạ trong 1 frame để suy luận ngữ cảnh hoàn chỉnh.
        
        Args:
            detections (list): Danh sách các dict [{'id': 1, 'class': 'Vehicle', 'bbox': [x1, y1, x2, y2]}]
            masks (dict): Chứa 'drivable' và 'lane' masks (2D numpy arrays)
            weather_label (str): Điều kiện thời tiết ("Nang rao", "Mua", "Ban dem", "Suong mu")
            frame_idx (int): Chỉ số frame hiện tại
            frame_shape (tuple): (height, width)
            
        Returns:
            dict: Cấu trúc JSON biểu diễn ngữ cảnh giao thông hoàn chỉnh
        """
        h, w = frame_shape[:2]
        masks = masks or {}
        drivable_mask = masks.get('drivable', None)
        lane_mask = masks.get('lane', None)

        # Ngưỡng TTC nhạy cảm hơn khi thời tiết xấu hoặc đêm tối
        ttc_threshold = 2.5 if weather_label in ["Mua", "Ban dem", "Suong mu"] else 1.8

        context_data = {
            "frame": frame_idx,
            "weather": weather_label,
            "summary": {
                "total_vehicles": 0,
                "total_pedestrians": 0,
                "pedestrians_on_road": 0,
                "traffic_light_ahead": False,
                "traffic_light_distance": None,
                "risk_level": "An toan",
                "active_warnings_count": 0
            },
            "warnings": [],
            "objects": []
        }

        vehicles_count = 0
        pedestrians_count = 0
        pedestrians_on_road = 0
        traffic_light_found = False
        traffic_light_dist = None
        max_risk_level = "An toan"

        for obj in detections:
            obj_id = obj.get('id', -1)
            obj_class = obj.get('class', 'Vehicle')
            bbox = obj.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            bottom_center = ((x1 + x2) / 2.0, float(y2))
            area = max(1, (x2 - x1) * (y2 - y1))

            distance = self.estimate_distance(bbox, (h, w), obj_class=obj_class)
            is_in_road = self.check_in_drivable(bottom_center, drivable_mask)
            lane_pos = self.determine_lane_position(bottom_center, lane_mask)

            ttc = float('inf')
            if obj_class == "Vehicle" and obj_id != -1:
                ttc = self.estimate_ttc(obj_id, area, frame_idx)

            risk_level = "Thap"

            # 1. Logic cho PHƯƠNG TIỆN (Vehicle)
            if obj_class == "Vehicle":
                vehicles_count += 1
                if is_in_road and (distance == "Gan" or (ttc != float('inf') and ttc < ttc_threshold)):
                    risk_level = "Cao"
                    ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "K/c Gan"
                    context_data["warnings"].append(
                        f"CANH BAO VA CHAM (FCW): Xe #{obj_id} nguy co cao (TTC: {ttc_str}, Lan: {lane_pos})"
                    )
                elif is_in_road and distance == "Vua":
                    risk_level = "Trung binh"

            # 2. Logic cho NGƯỜI ĐI BỘ (Pedestrian)
            elif obj_class == "Pedestrian":
                pedestrians_count += 1
                if is_in_road:
                    pedestrians_on_road += 1
                    risk_level = "Cao"
                    context_data["warnings"].append(
                        f"NGUY HIEM (PCW): Nguoi di bo tren long duong! (ID #{obj_id}, K/c: {distance})"
                    )
                elif distance == "Gan":
                    risk_level = "Trung binh"
                    context_data["warnings"].append(
                        f"CHU Y: Nguoi di bo gan mep duong (ID #{obj_id})"
                    )

            # 3. Logic cho ĐÈN GIAO THÔNG (Traffic Light)
            elif obj_class == "Traffic Light":
                traffic_light_found = True
                traffic_light_dist = distance
                if distance in ["Gan", "Vua"]:
                    context_data["warnings"].append(
                        f"TIN HIEU: Phat hien den giao thong phia truoc (Khoang cach: {distance})"
                    )

            # Cập nhật mức độ rủi ro chung của toàn frame
            if risk_level == "Cao":
                max_risk_level = "Nguy hiem"
            elif risk_level == "Trung binh" and max_risk_level != "Nguy hiem":
                max_risk_level = "Chu y"

            # Tạo chuỗi mô tả ngắn cho từng đối tượng (dùng hiển thị UI)
            desc_parts = [f"[{obj_class} #{obj_id}]", f"Lan: {lane_pos}", f"K/c: {distance}"]
            if is_in_road:
                desc_parts.append("Trong duong: Co")
            if ttc != float('inf'):
                desc_parts.append(f"TTC: {ttc:.1f}s")
            desc_parts.append(f"Rui ro: {risk_level}")
            short_desc = " | ".join(desc_parts)

            context_data["objects"].append({
                "id": obj_id,
                "class": obj_class,
                "bbox": bbox,
                "lane_position": lane_pos,
                "distance": distance,
                "in_drivable": bool(is_in_road),
                "ttc": round(ttc, 2) if ttc != float('inf') else None,
                "risk_level": risk_level,
                "description": short_desc
            })

        # Cập nhật block summary tổng kết
        context_data["summary"]["total_vehicles"] = vehicles_count
        context_data["summary"]["total_pedestrians"] = pedestrians_count
        context_data["summary"]["pedestrians_on_road"] = pedestrians_on_road
        context_data["summary"]["traffic_light_ahead"] = traffic_light_found
        context_data["summary"]["traffic_light_distance"] = traffic_light_dist
        context_data["summary"]["risk_level"] = max_risk_level
        context_data["summary"]["active_warnings_count"] = len(context_data["warnings"])

        return context_data