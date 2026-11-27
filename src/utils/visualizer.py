"""
Module Trực Quan Hóa ADAS & Ngữ Cảnh Giao Thông (ADAS Visualizer)
Tách rời độc lập theo chuẩn Clean Architecture (View Layer):
- Drivable Area Mask (Hòa trộn Emerald Green chống lóa, zero-allocation caching)
- Lane Markings (Vạch kẻ đường Neon Cyan chuẩn nét)
- YOLO 3D Bounding Boxes & Ground Contact Points
- Context Reasoning Spatial Graph (Vectơ liên kết quan hệ không gian từ Ego Vehicle)
- Chống xung đột / đè nhãn tự động khi mật độ phương tiện dày đặc
"""

import cv2
import numpy as np


class ADASVisualizer:
    """
    Bộ trực quan hóa ngữ cảnh giao thông (Traffic Context HUD) theo 4 cấp độ màu ngữ nghĩa:
    1. Emerald Green : Vùng mặt đường an toàn (Drivable Area) & Tín hiệu đèn Xanh
    2. Nominal Cyan  : Phương tiện theo dõi an toàn (Safe Tracked) & Vạch phân làn bán trong suốt
    3. Advisory Amber: Cảnh báo cấp 1 (Chuyển làn / Giảm khoảng cách / Đèn Vàng)
    4. Crimson Red   : Cảnh báo cấp 2 nguy hiểm cao (FCW / Người đi bộ dưới lòng đường / Đèn Đỏ)
    """

    # Bảng 4 màu ngữ nghĩa (Hệ màu BGR cho OpenCV)
    COLOR_GREEN_DRIVABLE = (95, 205, 46)    # Emerald Green cho mặt đường
    COLOR_GREEN_SIGNAL   = (60, 220, 40)    # Xanh lá tín hiệu đèn
    COLOR_CYAN_NOMINAL   = (245, 220, 0)    # Cyan chuẩn cho đối tượng an toàn
    COLOR_CYAN_LANE      = (255, 238, 95)   # Ice-Cyan dịu cho vạch kẻ làn bán trong suốt
    COLOR_AMBER_CAUTION  = (0, 165, 255)    # Cam Hổ Phách cho cảnh báo chú ý / chuyển làn
    COLOR_RED_DANGER     = (35, 25, 255)    # Đỏ Crimson cho cảnh báo nguy hiểm cao (FCW / PCW)

    def __init__(self):
        self._buffer_shape = None
        self._green_layer = None
        self._lane_layer = None
        self._blend_buf = None
        self._d_mask_buf = None
        self._l_mask_buf = None

    def draw_hud_visuals(self, frame, detections, masks, context_log, 
                         show_masks=True, show_lanes=True, show_boxes=True, 
                         show_graph=True, show_hud=True):
        """
        Vẽ trực quan hóa phong cách HUD ADAS Cockpit chuẩn Quốc tế SAE J2395:
        - SegFormer Drivable Area: Lớp phủ Emerald Green bán trong suốt (32% opacity)
        - SegFormer Lane Markings: Vạch kẻ làn Ice-Cyan bán trong suốt (58% opacity, không đè bệt mặt đường)
        - Context Reasoning Graph: Chỉ khóa tia thông minh tới Lead Vehicle & mục tiêu có rủi ro (chống rối mạng nhện)
        - YOLO Bounding Boxes: Hộp bao 4 màu chuẩn quốc tế + rút gọn nhãn thông minh cho xe ở xa
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Ưu tiên lấy danh sách đối tượng đã qua Context Engine suy luận
        render_objs = (
            context_log.get("objects")
            if isinstance(context_log, dict) and context_log.get("objects")
            else detections
        )

        # Quản lý bộ đệm màu định hình một lần (Zero-Allocation Buffer Caching)
        if self._buffer_shape != (h, w):
            self._green_layer = np.full((h, w, 3), self.COLOR_GREEN_DRIVABLE, dtype=np.uint8)
            self._lane_layer = np.full((h, w, 3), self.COLOR_CYAN_LANE, dtype=np.uint8)
            self._blend_buf = np.empty((h, w, 3), dtype=np.uint8)
            self._d_mask_buf = np.empty((h, w), dtype=np.uint8)
            self._l_mask_buf = np.empty((h, w), dtype=np.uint8)
            self._buffer_shape = (h, w)

        # 1. SegFormer Drivable Area (Hòa trộn Emerald Green 32% dịu mắt, giữ rõ chi tiết mặt nhựa đường)
        drivable = masks.get('drivable') if (show_masks and masks) else None
        lane = masks.get('lane') if (show_lanes and masks) else None
        y_roi = int(h * 0.25)

        if drivable is not None:
            if drivable.shape[:2] != (h, w):
                drivable = cv2.resize(drivable, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.addWeighted(
                annotated[y_roi:], 0.68,
                self._green_layer[y_roi:], 0.32,
                0, dst=self._blend_buf[y_roi:]
            )
            np.copyto(self._d_mask_buf[y_roi:], drivable[y_roi:])
            if render_objs:
                for obj in render_objs:
                    bx1, by1, bx2, by2 = obj['bbox']
                    y_s, y_e = max(y_roi, by1), min(h, by2)
                    x_s, x_e = max(0, bx1), min(w, bx2)
                    if y_e > y_s and x_s < x_e:
                        self._d_mask_buf[y_s:y_e, x_s:x_e] = 0
            cv2.copyTo(self._blend_buf[y_roi:], self._d_mask_buf[y_roi:], annotated[y_roi:])

        # 2. SegFormer Lane Markings (Hòa trộn bán trong suốt 58% để tách biệt hoàn toàn với khung BBox xe)
        if lane is not None:
            if lane.shape[:2] != (h, w):
                lane = cv2.resize(lane, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.addWeighted(
                annotated[y_roi:], 0.42,
                self._lane_layer[y_roi:], 0.58,
                0, dst=self._blend_buf[y_roi:]
            )
            np.copyto(self._l_mask_buf[y_roi:], lane[y_roi:])
            if render_objs:
                for obj in render_objs:
                    bx1, by1, bx2, by2 = obj['bbox']
                    y_s, y_e = max(y_roi, by1), min(h, by2)
                    x_s, x_e = max(0, bx1), min(w, bx2)
                    if y_e > y_s and x_s < x_e:
                        self._l_mask_buf[y_s:y_e, x_s:x_e] = 0
            cv2.copyTo(self._blend_buf[y_roi:], self._l_mask_buf[y_roi:], annotated[y_roi:])

        # Xác định Xe dẫn đầu chính & Đèn Giao Thông chính (để vẽ 1 nhãn duy nhất cho giàn đèn ngã tư)
        primary_lead_id = None
        primary_tl_id = None
        if render_objs:
            ego_candidates = [
                o for o in render_objs
                if o.get("class", "Vehicle") == "Vehicle"
                and any(k in str(o.get("lane_position", "")).lower() for k in ["cung lan", "giua", "ego"])
            ]
            if ego_candidates:
                best_lead = max(ego_candidates, key=lambda o: o["bbox"][3])
                primary_lead_id = best_lead.get("id")

            tl_candidates = [o for o in render_objs if o.get("class") == "Traffic Light"]
            if tl_candidates:
                # Chọn cụm đèn gần trục nhìn trung tâm xe chủ nhất để gắn nhãn đại diện
                best_tl = min(tl_candidates, key=lambda o: abs((o['bbox'][0] + o['bbox'][2]) // 2 - w // 2))
                primary_tl_id = best_tl.get("id")

        # 3. Context Reasoning Graph (Chỉ vẽ tia khóa mục tiêu trọng tâm, loại bỏ chùm tia mạng nhện cho xe ở xa)
        ego_origin = (int(w * 0.50), int(h * 0.90))
        if show_graph and render_objs:
            for obj in render_objs:
                cls_name = obj.get("class", "Vehicle")
                if cls_name == "Traffic Light":
                    continue
                x1, y1, x2, y2 = obj['bbox']
                risk_lvl = obj.get("risk_level", "Thap")
                is_emergency_obj = (risk_lvl == "Khan cap") or (cls_name == "Pedestrian" and obj.get("in_drivable"))
                is_cut_in_obj = obj.get("is_cut_in", False)
                is_caution_obj = is_cut_in_obj or obj.get("is_amber", False) or (risk_lvl in ["Cao", "Trung binh"])
                is_lead = obj.get("is_lead", False) or (obj.get("id") is not None and obj.get("id") == primary_lead_id)
                raw_dist = str(obj.get("distance", "Vua")).lower()
                is_near_or_mid = raw_dist in ["gan", "near", "vua", "mid", "medium"]

                # Chỉ vẽ tia cho: Xe dẫn đầu, Mục tiêu cảnh báo (Cam/Đỏ), hoặc Xe ở cự ly Gần/Vừa (bỏ xe an toàn ở xa)
                if not (is_emergency_obj or is_caution_obj or is_lead or (is_near_or_mid and len(render_objs) <= 4)):
                    continue

                target_pt = ((x1 + x2) // 2, y2)
                if is_emergency_obj:
                    line_color = self.COLOR_RED_DANGER
                    line_thick = 2
                elif is_caution_obj:
                    line_color = self.COLOR_AMBER_CAUTION
                    line_thick = 2
                else:
                    line_color = self.COLOR_CYAN_NOMINAL
                    line_thick = 1 if not is_lead else 2
                cv2.line(annotated, ego_origin, target_pt, line_color, line_thick, cv2.LINE_AA)

        # 4. YOLO Bounding Boxes & Nhãn Ngữ Cảnh Chuẩn 4 Màu Quốc Tế SAE J2395
        if show_boxes and render_objs:
            is_fhd = (w >= 1600)
            placed_label_rects = []
            many_objects = (len(render_objs) >= 5)
            # Ưu tiên vẽ xe gần trước (y2 lớn hơn)
            sorted_objs = sorted(render_objs, key=lambda o: o['bbox'][3], reverse=True)
            for obj in sorted_objs:
                x1, y1, x2, y2 = obj['bbox']
                risk_lvl = obj.get("risk_level", "Thap")
                is_emergency_obj = (risk_lvl == "Khan cap")
                is_cut_in_obj = obj.get("is_cut_in", False)
                is_caution_obj = is_cut_in_obj or obj.get("is_amber", False) or (risk_lvl in ["Cao", "Trung binh"])
                tl_col = obj.get("traffic_light_color", "Unknown")
                is_lead = obj.get("is_lead", False) or (obj.get("id") is not None and obj.get("id") == primary_lead_id)
                cls_name = obj.get("class", "Vehicle")
                custom_label = obj.get("label", None)

                # Quy chiếu nghiêm ngặt về đúng 4 Màu Ngữ Nghĩa Quốc Tế (Green - Cyan - Amber - Red)
                if cls_name == "Traffic Light":
                    if tl_col == "Red":
                        color = self.COLOR_RED_DANGER
                    elif tl_col == "Yellow":
                        color = self.COLOR_AMBER_CAUTION
                    elif tl_col == "Green":
                        color = self.COLOR_GREEN_SIGNAL
                    else:
                        color = self.COLOR_CYAN_NOMINAL
                elif is_emergency_obj or (cls_name == "Pedestrian" and obj.get("in_drivable")):
                    color = self.COLOR_RED_DANGER
                elif is_caution_obj:
                    color = self.COLOR_AMBER_CAUTION
                else:
                    color = self.COLOR_CYAN_NOMINAL

                # Độ dày viền: Đèn giao thông dùng viền mảnh 1-2px để không che lấp bóng đèn
                if cls_name == "Traffic Light":
                    box_thick = 1 if (w < 1600 or (x2 - x1 < 32)) else 2
                elif color == self.COLOR_RED_DANGER:
                    box_thick = 3
                else:
                    box_thick = 2

                # Vẽ khung Bounding Box khử răng cưa sắc nét
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick, cv2.LINE_AA)

                # Vẽ điểm tiếp xúc mặt đường (Ground Contact Point)
                if cls_name != "Traffic Light":
                    cx = (x1 + x2) // 2
                    r_in = 5 if is_fhd else 4
                    r_out = 8 if is_fhd else 6
                    cv2.circle(annotated, (cx, y2), r_in, color, -1, cv2.LINE_AA)
                    cv2.circle(annotated, (cx, y2), r_out, (255, 255, 255), 1, cv2.LINE_AA)

                # Nhãn hiển thị chuẩn Quốc tế: Đầy đủ cho xe chính/rủi ro, gọn gàng cho xe an toàn ở xa
                if custom_label:
                    label = custom_label
                else:
                    raw_lane = str(obj.get("lane_position", "Cung lan")).lower()
                    if "trai" in raw_lane or "left" in raw_lane:
                        short_lane = "Left Lane"
                    elif "phai" in raw_lane or "right" in raw_lane:
                        short_lane = "Right Lane"
                    else:
                        short_lane = "Ego Lane"

                    raw_dist = str(obj.get("distance", "Vua")).lower()
                    if raw_dist in ["gan", "near"]:
                        dist_en = "Near"
                    elif raw_dist in ["vua", "mid", "medium"]:
                        dist_en = "Mid"
                    elif raw_dist in ["xa", "far"]:
                        dist_en = "Far"
                    else:
                        dist_en = "Mid"

                    if is_emergency_obj:
                        label = f"[CRITICAL #{obj['id']} | {short_lane} | Dist: {dist_en}]"
                    elif is_cut_in_obj:
                        label = f"[LANE CHANGE #{obj['id']} | {short_lane} | Dist: {dist_en}]"
                    elif cls_name == "Vehicle":
                        # Nếu khung hình đông xe và xe ở xa an toàn, rút gọn nhãn để không che bầu trời
                        if many_objects and dist_en == "Far" and not is_caution_obj and not is_lead:
                            label = f"[#{obj['id']} | {short_lane}]"
                        else:
                            label = f"[Vehicle #{obj['id']} | {short_lane} | Dist: {dist_en}]"
                    elif cls_name == "Pedestrian":
                        ped_status = "ON ROAD!" if obj.get("in_drivable") else f"Sidewalk | {dist_en}"
                        label = f"[Pedestrian #{obj['id']} | {ped_status}]"
                    elif cls_name == "Traffic Light":
                        # Hiển thị nhãn ngắn gọn [RED] / [GREEN] / [YELLOW] trên TỪNG cụm đèn
                        # (xử lý chuẩn thực tế khi trên cùng giàn có cả đèn rẽ Xanh lẫn đèn đi thẳng Đỏ)
                        tl_state = tl_col.upper() if tl_col in ["Red", "Yellow", "Green"] else "SIGNAL"
                        label = f"[{tl_state}]"
                    else:
                        label = f"[{cls_name} #{obj['id']} | Dist: {dist_en}]"

                # Thanh tiêu đề trên đầu BBox (tự động chống đè chữ và không vẽ nếu label=None)
                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    is_compact_far = (many_objects and cls_name == "Vehicle" and not is_caution_obj and not is_emergency_obj and not is_lead and "Dist:" not in label)
                    base_scale = 0.58 if is_fhd else (0.48 if w >= 1200 else 0.42)
                    if cls_name == "Traffic Light":
                        font_scale = 0.44 if is_fhd else 0.36
                        font_thick = 1
                        pad_x = 4
                        pad_y = 6
                    else:
                        font_scale = base_scale * 0.88 if is_compact_far else base_scale
                        font_thick = 2 if (is_fhd and not is_compact_far) else 1
                        pad_x = 6 if is_fhd else 5
                        pad_y = 9 if is_fhd else 7
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
                    box_h_label = th + pad_y
                    if cls_name == "Traffic Light":
                        cx_tl = (x1 + x2) // 2
                        x_box_left = max(0, min(cx_tl - (tw + pad_x * 2) // 2, w - tw - pad_x * 2 - 2))
                    else:
                        x_box_left = max(0, min(x1 - 1, w - tw - pad_x * 2 - 2))
                    x_box_right = min(w - 1, x_box_left + tw + pad_x * 2)

                    if y1 >= box_h_label + 4:
                        y_box_top = y1 - box_h_label
                    else:
                        y_box_top = y1
                    y_box_bot = y_box_top + box_h_label

                    # Dịch chuyển nhãn lên trên nếu bị trùng (overlap) với nhãn trước đó
                    for _ in range(3):
                        conflict = False
                        for (rx1, ry1, rx2, ry2) in placed_label_rects:
                            if not (x_box_right < rx1 - 2 or x_box_left > rx2 + 2 or y_box_bot < ry1 - 2 or y_box_top > ry2 + 2):
                                conflict = True
                                if ry1 - box_h_label - 3 >= 2:
                                    y_box_top = ry1 - box_h_label - 3
                                else:
                                    y_box_top = min(h - box_h_label - 2, ry2 + 3)
                                y_box_bot = y_box_top + box_h_label
                                break
                        if not conflict:
                            break

                    placed_label_rects.append((x_box_left, y_box_top, x_box_right, y_box_bot))
                    y_text = y_box_bot - (5 if is_fhd else 4)

                    # Chữ trắng sắc nét trên nền Đỏ, chữ đen đậm trên nền Cyan / Amber / Green theo chuẩn tương phản WCAG
                    text_color = (255, 255, 255) if (color == self.COLOR_RED_DANGER) else (10, 12, 18)
                    cv2.rectangle(annotated, (x_box_left, y_box_top), (x_box_right, y_box_bot), color, -1, cv2.LINE_AA)
                    cv2.putText(annotated, label, (x_box_left + pad_x, y_text),
                                font, font_scale, text_color, font_thick, cv2.LINE_AA)

        return annotated
