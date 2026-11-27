"""
Module Suy Luận Ngữ Cảnh Giao Thông (Traffic Context Representation Engine)

Nhận diện và biểu diễn ngữ cảnh các nhân tố ảnh hưởng đến hoạt động giao thông.

"""

import numpy as np
import cv2


class VehicleTrajectoryTracker:
    """
    Bộ theo dõi chuyển động không-thời gian (Spatio-Temporal Tracker)
    - Quản lý lịch sử tọa độ (FIFO Buffer, tối đa 25 khung hình)
    - Khử rung lắc BBox bằng bộ lọc mượt hàm mũ (EMA)
    - Ước lượng vận tốc tương đối vx, vy (pixel/giây) và trạng thái chuyển động
    - Phát hiện xe chuyển từ làn bên vào làn chính (Lane Change Detection)
    - Tự động dọn dẹp các ID xe đã rời khỏi khung hình
    """

    def __init__(self, fps=30, max_history=25):
        self.fps = max(1.0, float(fps))
        self.max_history = max_history
        self.tracks = {}

    def reset(self):
        """Xóa sạch bộ đệm theo dõi khi nạp video mới"""
        self.tracks.clear()

    def update(self, track_id, bottom_center, area, lane_pos, distance, frame_idx, frame_shape):
        """
        Cập nhật vị trí mới của vật thể và tính toán quỹ đạo / vận tốc / dự báo tương lai.
        """
        if track_id is None or track_id == -1:
            return {
                "trajectory": [],
                "predicted_trajectory": [],
                "velocity": {"vx": 0.0, "vy": 0.0, "speed": 0.0},
                "motion_state": "Không xác định",
                "is_cut_in": False,
                "cut_in_reason": "",
                "smoothed_ttc": float('inf')
            }

        h_f, w_f = frame_shape[:2]
        x_curr, y_curr = float(bottom_center[0]), float(bottom_center[1])

        # Dọn dẹp các track đã biến mất quá 30 frames (Garbage Collection)
        dead_ids = [tid for tid, tdata in self.tracks.items() if frame_idx - tdata.get("last_frame", frame_idx) > 30]
        for tid in dead_ids:
            del self.tracks[tid]

        if track_id not in self.tracks:
            # Khởi tạo bản ghi mới
            self.tracks[track_id] = {
                "history": [(x_curr, y_curr, area, frame_idx)],
                "vx": 0.0,
                "vy": 0.0,
                "last_frame": frame_idx,
                "last_lane": lane_pos,
                "motion_state": "Đang theo dõi",
                "is_cut_in": False,
                "cut_in_reason": "",
                "smoothed_ttc": float('inf')
            }
            return {
                "trajectory": [(int(x_curr), int(y_curr))],
                "predicted_trajectory": [],
                "velocity": {"vx": 0.0, "vy": 0.0, "speed": 0.0},
                "motion_state": "Đang theo dõi",
                "is_cut_in": False,
                "cut_in_reason": "",
                "smoothed_ttc": float('inf')
            }

        t_data = self.tracks[track_id]
        history = t_data["history"]
        prev_x, prev_y, prev_area, prev_f = history[-1]
        delta_f = max(1, frame_idx - prev_f)
        dt = delta_f / self.fps

        # 1. Tính toán vận tốc tức thời thô và làm mượt bằng EMA
        raw_vx = (x_curr - prev_x) / dt
        raw_vy = (y_curr - prev_y) / dt

        alpha_v = 0.35
        vx = alpha_v * raw_vx + (1.0 - alpha_v) * t_data["vx"]
        vy = alpha_v * raw_vy + (1.0 - alpha_v) * t_data["vy"]
        speed = float(np.hypot(vx, vy))

        t_data["vx"] = vx
        t_data["vy"] = vy
        t_data["last_frame"] = frame_idx

        # Cập nhật bộ đệm lịch sử
        history.append((x_curr, y_curr, area, frame_idx))
        if len(history) > self.max_history:
            history.pop(0)

        # 2. Phân loại trạng thái động học (Motion State)
        if vy > 40.0 and distance in ["Gan", "Vua"]:
            motion_state = "Đang áp sát nhanh (Closing In)"
        elif vy < -30.0:
            motion_state = "Đang tách xa (Moving Away)"
        elif vx > w_f * 0.035:
            motion_state = "Chuyển làn phải (Swerving Right)"
        elif vx < -w_f * 0.035:
            motion_state = "Chuyển làn trái (Swerving Left)"
        else:
            motion_state = "Tốc độ ổn định (Cruising)"
        t_data["motion_state"] = motion_state

        # 3. Phát hiện chuyển làn trực tiếp vào làn chính (Lane Change Detection)
        is_cut_in = False
        cut_in_reason = ""
        prev_lane = t_data.get("last_lane", lane_pos)
        if ("lan trai" in prev_lane.lower() or "lan phai" in prev_lane.lower()) and ("cung lan" in lane_pos.lower()):
            if distance in ["Gan", "Vua"]:
                is_cut_in = True
                cut_in_reason = f"Xe #{track_id} chuyển từ {prev_lane} vào làn chính"

        t_data["last_lane"] = lane_pos
        t_data["is_cut_in"] = is_cut_in
        t_data["cut_in_reason"] = cut_in_reason

        # 4. Làm mượt chỉ số TTC tương đối (Smoothed Relative TTC)
        smoothed_ttc = float('inf')
        if len(history) >= 2 and area > history[0][2]:
            old_pt = history[-min(len(history), 4)]
            d_area = area - old_pt[2]
            d_frames = frame_idx - old_pt[3]
            if d_frames > 0 and d_area > 0:
                rate = d_area / d_frames
                raw_ttc = (area / rate) / self.fps
                prev_ttc = t_data.get("smoothed_ttc", float('inf'))
                if prev_ttc != float('inf'):
                    smoothed_ttc = 0.45 * raw_ttc + 0.55 * prev_ttc
                else:
                    smoothed_ttc = raw_ttc
                t_data["smoothed_ttc"] = smoothed_ttc

        trajectory_pts = [(int(pt[0]), int(pt[1])) for pt in history]

        return {
            "trajectory": trajectory_pts,
            "predicted_trajectory": [],
            "velocity": {"vx": round(vx, 1), "vy": round(vy, 1), "speed": round(speed, 1)},
            "motion_state": motion_state,
            "is_cut_in": is_cut_in,
            "cut_in_reason": cut_in_reason,
            "smoothed_ttc": smoothed_ttc
        }


class TrafficContextEngine:
    """
    Bộ suy luận ngữ cảnh giao thông kết hợp đầu ra từ:
    - Object Detection + Tracking (YOLOv12 + ByteTrack): BBoxes, Class, Track ID
    - Semantic Segmentation (SegFormer): Mask Drivable Area & Lane Markings
    """

    def __init__(self, fps=30):
        self.fps = fps
        self.prev_boxes = {}   # Lưu trữ: track_id -> (area, frame_idx)
        self.tracker = VehicleTrajectoryTracker(fps=fps, max_history=25)
        self._cached_env_info = None
        self._last_env_frame = -999
        self._cached_vp = (None, None)
        self._last_drivable_id = None

    def reset(self):
        """Đặt lại trạng thái các bộ đệm khi bắt đầu video mới"""
        self.prev_boxes.clear()
        self.tracker.reset()
        self._cached_env_info = None
        self._last_env_frame = -999
        self._cached_vp = (None, None)
        self._last_drivable_id = None

    def assess_environment_and_camera(self, frame=None, user_weather="Tu dong", frame_idx=0):
        """
        Đánh giá hoàn toàn tự động chất lượng thấu kính camera và điều kiện thời tiết/môi trường.
        Tối ưu hóa hiệu năng: Cache kết quả mỗi 30 frames (1 giây) và tính trên ảnh thu nhỏ 160x90.
        """
        if self._cached_env_info is not None and abs(frame_idx - self._last_env_frame) < 30:
            return self._cached_env_info

        blur_var = 180.0
        mean_bright = 128.0
        std_bright = 50.0
        is_rain = False
        is_fog = False

        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            try:
                small_f = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA) if frame.shape[1] > 200 else frame
                if len(small_f.shape) == 3:
                    gray = cv2.cvtColor(small_f, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(small_f, cv2.COLOR_BGR2HSV)
                else:
                    gray = small_f
                    hsv = None

                gh, gw = gray.shape[:2]
                roi_center = gray[int(gh * 0.2):int(gh * 0.8), int(gw * 0.15):int(gw * 0.85)]
                blur_var = float(cv2.Laplacian(roi_center, cv2.CV_64F).var())
                mean_bright = float(np.mean(roi_center))
                std_bright = float(np.std(roi_center))

                # Phân tích mặt đường (40% nửa dưới khung hình) tìm phản xạ gương nước (Wet Road Specular)
                road_roi = gray[int(gh * 0.6):gh, int(gw * 0.2):int(gw * 0.8)]
                road_p98 = float(np.percentile(road_roi, 98))
                road_p10 = float(np.percentile(road_roi, 10))
                road_mean = float(np.mean(road_roi))
                road_contrast = road_p98 - road_p10

                # Phân tích độ bão hòa màu bầu trời/quang cảnh (HSV Saturation)
                if hsv is not None:
                    sky_roi_sat = hsv[0:int(gh * 0.4), :, 1]
                    mean_sky_sat = float(np.mean(sky_roi_sat))
                else:
                    mean_sky_sat = 55.0

                # Nhận diện mưa bão / đường ướt:
                # 1. Đường ướt có độ tương phản cực đoan (vệt đèn/bầu trời phản chiếu trên nhựa đường tối)
                # 2. Hoặc bầu trời xám xịt âm u (sat thấp) kèm giọt nước mờ kính (blur_var < 95)
                if (road_contrast > 140.0 and road_p98 > 190.0 and road_mean < 115.0) or \
                   (mean_sky_sat < 36.0 and 55.0 <= mean_bright <= 140.0 and blur_var < 95.0):
                    is_rain = True

                # Nhận diện sương mù / khói mù quang học:
                # Toàn cảnh bị mờ nhạt, tương phản thấp (std_bright < 32), không có góc cạnh sắc (blur_var < 70)
                if std_bright < 32.0 and blur_var < 70.0 and mean_bright > 65.0:
                    is_fog = True

            except Exception:
                pass

        # 1. Tình trạng vật lý của ống kính camera
        if blur_var < 55.0:
            camera_status = "Mo ro ret (Occlusion / Giot nuoc / Bui)"
            is_camera_compromised = True
        elif blur_var < 105.0:
            camera_status = "Hoi mo (Suong mu / Mua nho)"
            is_camera_compromised = True
        else:
            camera_status = "Sac net (Clear)"
            is_camera_compromised = False

        # 2. Điều kiện ánh sáng
        if mean_bright < 50.0:
            lighting_status = "Ban dem / Thieu sang"
        elif mean_bright > 215.0:
            lighting_status = "Choi sang / Nguoc nang"
        else:
            lighting_status = "Ban ngay"

        # 3. Phân định thời tiết: TỰ ĐỘNG HOÀN TOÀN TỪ CAMERA
        w_lower = str(user_weather).lower() if user_weather else "tu dong"
        if any(k in w_lower for k in ["tu dong", "tự động", "auto", "optical"]) or not user_weather:
            if lighting_status == "Ban dem / Thieu sang":
                weather_final = "Ban đêm"
            elif is_rain:
                weather_final = "Mưa bão / Đường ướt"
            elif is_camera_compromised or is_fog:
                weather_final = "Sương mù / Kính mờ"
            else:
                weather_final = "Nắng ráo"
        else:
            weather_final = user_weather

        # 4. Điều chỉnh ngưỡng cảnh báo tương đối theo điều kiện thời tiết
        wf_lower = weather_final.lower()
        if any(k in wf_lower for k in ["mua", "bao", "rain", "storm", "uon"]):
            ttc_threshold = 3.2
            weather_advisory = "Thời tiết mưa / đường ướt - Tăng khoảng cách an toàn"
        elif any(k in wf_lower for k in ["suong", "fog", "cam mo", "mo"]):
            ttc_threshold = 2.9
            weather_advisory = "Tầm nhìn hạn chế do sương mù/kính mờ - Giảm tốc độ"
        elif any(k in wf_lower for k in ["dem", "night", "toi"]):
            ttc_threshold = 2.4
            weather_advisory = "Thiếu sáng ban đêm - Tăng cường quan sát"
        else:
            ttc_threshold = 1.8
            weather_advisory = "Điều kiện thời tiết bình thường"

        res_env = {
            "weather": weather_final,
            "camera_status": camera_status,
            "is_camera_compromised": is_camera_compromised,
            "blur_variance": round(blur_var, 1),
            "lighting": lighting_status,
            "brightness": round(mean_bright, 1),
            "ttc_threshold": ttc_threshold,
            "advisory": weather_advisory
        }
        self._cached_env_info = res_env
        self._last_env_frame = frame_idx
        return res_env

    def detect_traffic_light_color(self, frame, bbox):
        """
        Phân tích vùng quan tâm (ROI) của đèn giao thông trong không gian màu HSV
        để xác định trạng thái tín hiệu: RED (Dừng), YELLOW (Giảm tốc), GREEN (Được đi).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return "Unknown"

        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w_f - 1, x1))
        y1 = max(0, min(h_f - 1, y1))
        x2 = max(0, min(w_f, x2))
        y2 = max(0, min(h_f, y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 3 or bh < 6:
            return "Unknown"

        roi = frame[y1:y2, x1:x2]
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Mặt nạ màu ĐỎ (bao phủ cả 2 đầu dải Hue 0-10 và 160-180)
            mask_r1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
            mask_r2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)

            # Mặt nạ màu VÀNG
            mask_yellow = cv2.inRange(hsv, np.array([15, 75, 75]), np.array([36, 255, 255]))

            # Mặt nạ màu XANH LÁ
            mask_green = cv2.inRange(hsv, np.array([42, 60, 60]), np.array([92, 255, 255]))

            cnt_red = int(cv2.countNonZero(mask_red))
            cnt_yellow = int(cv2.countNonZero(mask_yellow))
            cnt_green = int(cv2.countNonZero(mask_green))

            max_cnt = max(cnt_red, cnt_yellow, cnt_green)
            min_pixels = max(4, int(bw * bh * 0.05))

            if max_cnt >= min_pixels:
                if max_cnt == cnt_red:
                    return "Red"
                elif max_cnt == cnt_yellow:
                    return "Yellow"
                elif max_cnt == cnt_green:
                    return "Green"
        except Exception:
            pass

        return "Unknown"

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
            if rel_height >= 0.055:
                return "Gan"
            elif rel_height >= 0.022:
                return "Vua"
            else:
                return "Xa"
        else:
            rel_y = y2 / h
            rel_h = (y2 - y1) / h
            if rel_y > 0.74 or rel_h > 0.30:
                return "Gan"
            elif rel_y > 0.56 or rel_h > 0.14:
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

    def determine_lane_position(self, bottom_point, lane_mask, drivable_mask=None, bbox=None):
        """
        Xác định vị trí làn xe tương đối so với xe Ego (Cung lan / Lan trai / Lan phai)
        kết hợp mô hình nón phối cảnh thích ứng mặt đường (Dynamic Drivable Corridor),
        độ phủ ngang hộp bao (BBox Horizontal Overlap) và vạch kẻ làn SegFormer.
        """
        x, y = int(bottom_point[0]), int(bottom_point[1])
        ref_mask = lane_mask if (lane_mask is not None and lane_mask.size > 0) else drivable_mask
        if ref_mask is None or ref_mask.size == 0:
            return "Cung lan"

        h, w = ref_mask.shape[:2]
        if not (0 <= y < h and 0 <= x < w):
            return "Ngoai khung hinh"

        bw = float(bbox[2] - bbox[0]) if (bbox is not None and bbox[2] > bbox[0]) else (w * 0.06)
        x_bottom = w * 0.50

        # 1. Trích xuất hình học mặt đường từ drivable_mask (chỉ lấy lát cắt ngang siêu tốc)
        road_center = None
        road_w = None
        vp_x = None
        vp_y = None

        if drivable_mask is not None and drivable_mask.size > 0:
            y_min = max(0, y - 8)
            y_max = min(h, y + 9)
            patch = drivable_mask[y_min:y_max, :]
            col_hits = np.any(patch > 0, axis=0)
            xs = np.where(col_hits)[0]
            if len(xs) > 10:
                road_left = float(xs[0])
                road_right = float(xs[-1])
                road_w = road_right - road_left
                road_center = (road_left + road_right) / 2.0

            # Ước lượng Điểm tụ động từ 2 cạnh biên mặt đường (Cache theo đối tượng drivable_mask)
            if id(drivable_mask) == self._last_drivable_id:
                vp_x, vp_y = self._cached_vp
            else:
                left_pts = []
                right_pts = []
                step_y = max(16, int(h * 0.05))
                for y_samp in range(int(h * 0.50), int(h * 0.90), step_y):
                    row = np.where(drivable_mask[y_samp, :] > 0)[0]
                    if len(row) > int(w * 0.05):
                        left_pts.append((float(row[0]), float(y_samp)))
                        right_pts.append((float(row[-1]), float(y_samp)))

                if len(left_pts) >= 4 and len(right_pts) >= 4:
                    lx, ly = zip(*left_pts)
                    rx, ry = zip(*right_pts)
                    try:
                        ml, bl = np.polyfit(ly, lx, 1)
                        mr, br = np.polyfit(ry, rx, 1)
                        denom = ml - mr
                        if abs(denom) > 1e-4:
                            cand_vp_y = (br - bl) / denom
                            cand_vp_x = ml * cand_vp_y + bl
                            if -h * 0.2 <= cand_vp_y <= h * 0.65 and -w * 0.2 <= cand_vp_x <= w * 1.2:
                                if road_w is not None and road_w > w * 0.52:
                                    vp_x = max(w * 0.46, min(w * 0.56, cand_vp_x))
                                else:
                                    vp_x = max(w * 0.36, min(w * 0.64, cand_vp_x))
                                vp_y = max(h * 0.32, min(h * 0.58, cand_vp_y))
                    except Exception:
                        pass
                self._last_drivable_id = id(drivable_mask)
                self._cached_vp = (vp_x, vp_y)

        # 2. Xác định tâm hành lang Ego tại y (ego_path_x) và bề rộng làn theo phối cảnh chuẩn (Perspective Lane Width)
        y_ref = vp_y if vp_y is not None else (h * 0.40)
        t = max(0.0, min(1.0, (y - y_ref) / max(1.0, h - y_ref)))
        perspective_half_w = w * (0.024 + 0.146 * t)

        if vp_x is not None:
            ego_path_x = vp_x + (x_bottom - vp_x) * t
        elif road_center is not None and road_w is not None and road_w > 20:
            if road_w <= w * 0.45:
                ego_path_x = road_center
            else:
                ego_path_x = (w * 0.50) * 0.78 + road_center * 0.22
        else:
            ego_path_x = x_bottom

        half_ego_width = max(perspective_half_w, min(w * 0.14, bw * 0.52))
        ego_left_bound = ego_path_x - half_ego_width
        ego_right_bound = ego_path_x + half_ego_width

        # 3. Quét vạch kẻ làn SegFormer nếu có (neo biên làn theo vạch phân làn gần tâm Ego nhất)
        if lane_mask is not None and lane_mask.size > 0:
            y_top = max(0, y - 14)
            y_bot = min(h, y + 15)
            band = lane_mask[y_top:y_bot, :]
            col_hits = np.any(band > 0, axis=0)
            lane_indices = np.where(col_hits)[0]

            if len(lane_indices) > 0:
                lanes_x = []
                cluster = [lane_indices[0]]
                cluster_gap = max(10, int(w * 0.012))
                for idx in lane_indices[1:]:
                    if idx - cluster[-1] <= cluster_gap:
                        cluster.append(idx)
                    else:
                        lanes_x.append(int(np.mean(cluster)))
                        cluster = [idx]
                lanes_x.append(int(np.mean(cluster)))

                min_d = half_ego_width * 0.45
                max_d = half_ego_width * 1.45
                left_lines = [lx for lx in lanes_x if (ego_path_x - max_d) <= lx <= (ego_path_x - min_d)]
                right_lines = [lx for lx in lanes_x if (ego_path_x + min_d) <= lx <= (ego_path_x + max_d)]
                if left_lines:
                    ego_left_bound = float(max(left_lines))
                if right_lines:
                    ego_right_bound = float(min(right_lines))

        # 4. Kiểm tra tọa độ tâm chân xe và độ phủ ngang của Bounding Box với hành lang Ego
        if bbox is not None and bbox[2] > bbox[0]:
            bx1, bx2 = float(bbox[0]), float(bbox[2])
            overlap = max(0.0, min(bx2, ego_right_bound) - max(bx1, ego_left_bound))
            overlap_ratio = overlap / max(1.0, bx2 - bx1)
            if (ego_left_bound <= x <= ego_right_bound) or (overlap_ratio >= 0.48):
                return "Cung lan"

        if x < ego_left_bound:
            return "Lan trai"
        elif x > ego_right_bound:
            return "Lan phai"
        return "Cung lan"

    def process_frame(self, detections, masks=None, weather_label="Nang rao", frame_idx=0, frame_shape=(384, 640), raw_frame=None):
        """
        Hợp nhất đối tượng phát hiện (YOLO) và mặt nạ phân đoạn (SegFormer) để biểu diễn ngữ cảnh giao thông:
        - Nhận diện điều kiện thời tiết & ánh sáng.
        - Xác định vị trí làn đường, lòng đường và khoảng cách tương đối.
        - Phân tích màu đèn tín hiệu giao thông (Đỏ, Vàng, Xanh).
        - Phát cảnh báo va chạm phía trước (FCW) và người đi bộ trên lòng đường (PCW).
        """
        h, w = frame_shape[:2]
        masks = masks or {}
        drivable_mask = masks.get('drivable', None)
        lane_mask = masks.get('lane', None)

        # 1. Đánh giá Thời tiết & Ánh sáng từ khung hình
        env_info = self.assess_environment_and_camera(frame=raw_frame, user_weather=weather_label, frame_idx=frame_idx)
        active_weather = env_info["weather"]
        ttc_threshold = env_info["ttc_threshold"]
        camera_status = env_info["camera_status"]

        context_data = {
            "frame": frame_idx,
            "weather": active_weather,
            "environment": {
                "weather": active_weather,
                "camera_status": camera_status,
                "blur_variance": env_info["blur_variance"],
                "lighting": env_info["lighting"],
                "safety_margin": f"Ngưỡng TTC tương đối {ttc_threshold}s"
            },
            "summary": {
                "total_vehicles": 0,
                "total_pedestrians": 0,
                "pedestrians_on_road": 0,
                "traffic_light_ahead": False,
                "traffic_light_color": "None",
                "traffic_light_distance": None,
                "cut_in_detected": False,
                "emergency_brake": False,
                "risk_level": "An toan",
                "active_warnings_count": 0
            },
            "warnings": [],
            "objects": []
        }

        # Đưa khuyến nghị môi trường vào danh sách cảnh báo nếu có
        if env_info["is_camera_compromised"]:
            context_data["warnings"].append(
                f"📹 CẢNH BÁO TẦM NHÌN: Ống kính camera {camera_status.lower()} - Đề nghị quan sát cẩn trọng!"
            )
        if any(k in active_weather.lower() for k in ["mưa", "mua", "rain"]):
            context_data["warnings"].append(
                f"🌧️ CẢNH BÁO THỜI TIẾT: {env_info['advisory']}"
            )

        vehicles_count = 0
        pedestrians_count = 0
        pedestrians_on_road = 0
        traffic_light_found = False
        traffic_light_dist = None
        traffic_light_color_main = "None"
        cut_in_any = False
        emergency_reasons = []

        for obj in detections:
            obj_id = obj.get('id', -1)
            obj_class = obj.get('class', 'Vehicle')
            bbox = obj.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            bottom_center = ((x1 + x2) / 2.0, float(y2))
            area = max(1, (x2 - x1) * (y2 - y1))

            distance = self.estimate_distance(bbox, (h, w), obj_class=obj_class)
            is_in_road = self.check_in_drivable(bottom_center, drivable_mask)
            lane_pos = self.determine_lane_position(bottom_center, lane_mask, drivable_mask=drivable_mask, bbox=bbox)

            # Cập nhật Spatio-Temporal Trajectory Tracker
            track_info = self.tracker.update(
                track_id=obj_id,
                bottom_center=bottom_center,
                area=area,
                lane_pos=lane_pos,
                distance=distance,
                frame_idx=frame_idx,
                frame_shape=(h, w)
            ) if obj_class == "Vehicle" else {
                "trajectory": [(int(bottom_center[0]), int(bottom_center[1]))],
                "predicted_trajectory": [],
                "velocity": {"vx": 0.0, "vy": 0.0, "speed": 0.0},
                "motion_state": "Người đi bộ" if obj_class == "Pedestrian" else "Cố định",
                "is_cut_in": False,
                "cut_in_reason": "",
                "smoothed_ttc": float('inf')
            }

            ttc = float('inf')
            if obj_class == "Vehicle" and obj_id != -1:
                ttc = self.estimate_ttc(obj_id, area, frame_idx)
                # Khi xe có ít nhất 3 frame bám vết, ưu tiên lấy TTC đã khử nhiễu ổn định
                if track_info.get("smoothed_ttc", float('inf')) != float('inf') and len(track_info.get("trajectory", [])) >= 3:
                    ttc = track_info["smoothed_ttc"]
                # Chống nhiễu BBox ở cự ly Xa (Dist: Far > 35m): dao động vài pixel ở xa không thể gây TTC < 3.2s
                if distance == "Xa" and ttc != float('inf'):
                    ttc = max(ttc, 3.5)
                elif distance == "Vua" and y2 < h * 0.58 and ttc != float('inf'):
                    ttc = max(ttc, 1.8)

            risk_level = "Thap"
            is_cut_in = False
            cut_in_msg = ""
            tl_color = "Unknown"

            # -------------------------------------------------------------
            # 1. Logic cho PHƯƠNG TIỆN (Vehicle) & Phát hiện TẠT ĐẦU (Cut-in)
            # -------------------------------------------------------------
            is_same_lane = ("cung lan" in lane_pos.lower() or "giua" in lane_pos.lower())
            if obj_class == "Vehicle":
                vehicles_count += 1
                
                # Kiểm tra hành vi tạt đầu xe qua Spatio-Temporal Trajectory Tracker
                is_cut_in = track_info.get("is_cut_in", False)
                cut_in_msg = track_info.get("cut_in_reason", "")
                if is_cut_in:
                    cut_in_any = True
                    risk_level = "Cao"
                    context_data["warnings"].append(f"⚠️ CẢNH BÁO XE TẠT ĐẦU: {cut_in_msg}!")

                # Phân cấp rủi ro va chạm phía trước (FCW) ở cự ly Gần hoặc Vừa sát đầu xe
                if is_in_road and is_same_lane:
                    is_close_range = (distance == "Gan") or (distance == "Vua" and y2 > h * 0.62)
                    if (is_close_range and ttc != float('inf') and ttc < 1.3) or (distance == "Gan" and ttc != float('inf') and ttc < 1.6) or (y2 > h * 0.82):
                        risk_level = "Khan cap"
                        emergency_reasons.append(f"Xe #{obj_id} cùng làn ở cự ly rất gần (TTC: {ttc:.1f}s)")
                        ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "K/c Gan"
                        context_data["warnings"].append(
                            f"CẢNH BÁO VA CHẠM MỨC CAO (FCW): Xe #{obj_id} nguy cơ cao (TTC: {ttc_str}, Làn: {lane_pos})"
                        )
                    elif distance == "Gan" or (distance == "Vua" and ttc != float('inf') and ttc < ttc_threshold):
                        risk_level = "Cao"
                        ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "K/c Gan"
                        context_data["warnings"].append(
                            f"CẢNH BÁO VA CHẠM (FCW): Xe #{obj_id} nguy cơ cao (TTC: {ttc_str}, Làn: {lane_pos})"
                        )
                    elif distance == "Vua":
                        risk_level = "Trung binh"
                elif is_in_road and not is_same_lane:
                    if is_cut_in:
                        risk_level = "Cao"
                    elif distance == "Gan":
                        risk_level = "Trung binh"

            # -------------------------------------------------------------
            # 2. Logic cho NGƯỜI ĐI BỘ (Pedestrian - PCW)
            # -------------------------------------------------------------
            elif obj_class == "Pedestrian":
                pedestrians_count += 1
                if is_in_road:
                    pedestrians_on_road += 1
                    if distance == "Gan" or y2 > h * 0.65:
                        risk_level = "Khan cap"
                        emergency_reasons.append(f"Người đi bộ #{obj_id} trên lòng đường ở cự ly GẦN")
                        context_data["warnings"].append(
                            f"NGUY HIỂM MỨC CAO (PCW): Người đi bộ trên lòng đường! (ID #{obj_id}, K/c: {distance})"
                        )
                    else:
                        risk_level = "Cao"
                        context_data["warnings"].append(
                            f"NGUY HIỂM (PCW): Người đi bộ trên lòng đường! (ID #{obj_id}, K/c: {distance})"
                        )
                elif distance == "Gan":
                    risk_level = "Trung binh"
                    context_data["warnings"].append(
                        f"CHÚ Ý: Người đi bộ sát mép vỉa hè (ID #{obj_id})"
                    )

            # -------------------------------------------------------------
            # 3. Logic cho ĐÈN GIAO THÔNG (Traffic Light: Đỏ / Vàng / Xanh)
            # -------------------------------------------------------------
            elif obj_class == "Traffic Light":
                traffic_light_found = True
                tl_color = self.detect_traffic_light_color(raw_frame, bbox)
                traffic_light_color_main = tl_color
                if traffic_light_dist is None or distance == "Gan" or (distance == "Vua" and traffic_light_dist == "Xa"):
                    traffic_light_dist = distance

                if tl_color == "Red":
                    context_data["warnings"].append(
                        f"🛑 TÍN HIỆU ĐÈN ĐỎ: Phát hiện đèn ĐỎ phía trước ({distance}) - Chuẩn bị dừng xe an toàn!"
                    )
                    if distance in ["Gan", "Vua"] and risk_level != "Khan cap":
                        risk_level = "Cao"
                elif tl_color == "Yellow":
                    context_data["warnings"].append(
                        f"🟡 TÍN HIỆU ĐÈN VÀNG: Đèn tín hiệu chuyển VÀNG ({distance}) - Giảm tốc độ giao lộ!"
                    )
                    if risk_level != "Khan cap":
                        risk_level = "Trung binh"
                elif tl_color == "Green":
                    if not any("ĐÈN XANH" in w for w in context_data["warnings"]):
                        context_data["warnings"].append(f"🟢 TÍN HIỆU ĐÈN XANH: Tuyến đường thông suốt.")
                else:
                    if distance in ["Gan", "Vua"] and not any("giao thông" in w for w in context_data["warnings"]):
                        context_data["warnings"].append(f"TÍN HIỆU: Phát hiện đèn giao thông phía trước (Khoảng cách: {distance})")

            # Mô tả ngắn cho đối tượng
            desc_parts = [f"[{obj_class} #{obj_id}]", f"Làn: {lane_pos}", f"K/c: {distance}"]
            if is_cut_in:
                desc_parts.append("TẠT ĐẦU!")
            if is_in_road:
                desc_parts.append("Trong đường: Có")
            if ttc != float('inf'):
                desc_parts.append(f"TTC: {ttc:.1f}s")
            desc_parts.append(f"Rủi ro: {risk_level}")
            short_desc = " | ".join(desc_parts)

            context_data["objects"].append({
                "id": obj_id,
                "class": obj_class,
                "bbox": bbox,
                "lane_position": lane_pos,
                "distance": distance,
                "in_drivable": bool(is_in_road),
                "is_cut_in": bool(is_cut_in),
                "traffic_light_color": tl_color if obj_class == "Traffic Light" else None,
                "ttc": round(ttc, 2) if ttc != float('inf') else None,
                "risk_level": risk_level,
                "velocity": track_info.get("velocity", {"vx": 0.0, "vy": 0.0, "speed": 0.0}),
                "motion_state": track_info.get("motion_state", ""),
                "description": short_desc
            })

        # -------------------------------------------------------------
        # 4. ĐÁNH GIÁ TỔNG THỂ MỨC ĐỘ RỦI RO (RISK LEVEL)
        # -------------------------------------------------------------
        if len(emergency_reasons) > 0:
            final_risk = "DANGER"
            context_data["summary"]["emergency_brake"] = True
            context_data["warnings"].insert(
                0, f"🚨 CẢNH BÁO NGUY HIỂM CẤP ĐỘ CAO: {'; '.join(emergency_reasons[:2])}!"
            )
        elif any(o["risk_level"] in ["Cao", "Khan cap"] for o in context_data["objects"]) or cut_in_any:
            final_risk = "CAUTION"
        elif any(o["risk_level"] == "Trung binh" for o in context_data["objects"]):
            final_risk = "CAUTION"
        else:
            final_risk = "SAFE"

        # Cập nhật block summary tổng kết
        context_data["summary"]["total_vehicles"] = vehicles_count
        context_data["summary"]["total_pedestrians"] = pedestrians_count
        context_data["summary"]["pedestrians_on_road"] = pedestrians_on_road
        context_data["summary"]["traffic_light_ahead"] = traffic_light_found
        context_data["summary"]["traffic_light_color"] = traffic_light_color_main
        context_data["summary"]["traffic_light_distance"] = traffic_light_dist
        context_data["summary"]["cut_in_detected"] = cut_in_any
        context_data["summary"]["risk_level"] = final_risk
        context_data["summary"]["active_warnings_count"] = len(context_data["warnings"])

        return context_data