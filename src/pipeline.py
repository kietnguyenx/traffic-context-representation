"""
Module Pipeline Nhận Thức Lái Xe Đa Nhiệm (Driving Perception Pipeline)
Tích hợp:
1. Object Detection & Tracking (YOLOv12 + ByteTrack)
2. Semantic Segmentation (SegFormer: Drivable Area & Lane Markings)
3. Suy luận ngữ cảnh giao thông (Traffic Context Representation Engine)
4. Trực quan hóa chuẩn HUD ADAS (Overlay Mask, Bounding Boxes, Alert Banners)

Tác giả: Thành viên 3 (Phụ trách Context Engine, Pipeline & Web Demo)
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo đường dẫn module luôn trỏ đúng
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from src.context_engine import TrafficContextEngine
    from src.utils.segformer_loader import SegFormerWrapper
    from src.utils.visualizer import ADASVisualizer
except ImportError:
    from context_engine import TrafficContextEngine
    from utils.segformer_loader import SegFormerWrapper
    from utils.visualizer import ADASVisualizer


class DrivingPerceptionPipeline:
    def __init__(self, yolo_weights=None, segformer_weights=None, fps=30):
        """
        Khởi tạo Pipeline xử lý luồng nhận thức lái xe.
        
        Args:
            yolo_weights (str, optional): Đường dẫn tới file trọng số yolo_best.pt
            segformer_weights (str, optional): Đường dẫn tới file trọng số segformer_best.pth
            fps (int): Số khung hình trên giây của video
        """
        self.fps = fps
        self.engine = TrafficContextEngine(fps=fps)
        self.yolo_model = None
        self.yolo_weights_path = None
        self.segformer_wrapper = None
        self.segformer_weights_path = None
        self.use_mock_detection = False
        self.use_mock_segmentation = False
        self.last_masks = None
        self.last_detections = None
        self.ema_fps = float(fps)
        self.visualizer = ADASVisualizer()
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
        except Exception:
            self.has_cuda = False
        self.device_target = 0 if self.has_cuda else "cpu"

        # Tự động quét tìm file trọng số mặc định nếu không truyền
        resolved_yolo = yolo_weights or self._auto_find_weights(
            ["weights/yolo_best.pt", "src/weights/yolo_best.pt", "../weights/yolo_best.pt", "yolo_best.pt"]
        )
        resolved_seg = segformer_weights or self._auto_find_weights(
            [
                "weights/segformer_best.pth", "weights/segformer_best.pt",
                "src/weights/segformer_best.pth", "src/weights/segformer_best.pt",
                "weights/segformer_model", "../weights/segformer_best.pth",
                "segformer_best.pth"
            ]
        )

        # 1. Nạp mô hình YOLO Detection & Tracking (Thành viên 1)
        self.load_yolo_weights(resolved_yolo)

        # 2. Nạp mô hình SegFormer Semantic Segmentation (Thành viên 2)
        self.load_segformer_weights(resolved_seg)

    @staticmethod
    def _auto_find_weights(candidate_paths):
        """Tự động tìm kiếm file trọng số theo danh sách ưu tiên"""
        project_root = Path(__file__).resolve().parent.parent
        for p in candidate_paths:
            # Kiểm tra đường dẫn tuyệt đối hoặc tương đối theo cwd
            if os.path.exists(p):
                return p
            # Kiểm tra theo thư mục gốc dự án
            root_p = project_root / p
            if root_p.exists():
                return str(root_p)
        return None

    def load_yolo_weights(self, weights_path):
        """Nạp hoặc đổi file trọng số YOLOv12"""
        if weights_path and os.path.exists(weights_path):
            try:
                from ultralytics import YOLO
                print(f"[Pipeline] Nạp trọng số YOLO từ: {weights_path}")
                self.yolo_model = YOLO(weights_path)
                try:
                    self.yolo_model.fuse()
                except Exception:
                    pass
                self.yolo_weights_path = weights_path
                self.use_mock_detection = False
                return True
            except Exception as e:
                print(f"[Pipeline] Lỗi nạp YOLO từ {weights_path}: {e}. Chuyển sang Fallback.")
        
        # Không tự động tải lại yolo12n.pt mẫu về thư mục gốc nữa; chờ file weights/yolo_best.pt chính thức
        print("[Pipeline] Chưa tìm thấy weights/yolo_best.pt -> Hoạt động ở chế độ Detection Synthetic Mock.")
        self.yolo_model = None
        self.yolo_weights_path = None
        self.use_mock_detection = True
        return False

    def load_segformer_weights(self, weights_path):
        """Nạp hoặc đổi file trọng số SegFormer"""
        self.segformer_wrapper = SegFormerWrapper(weights_path=weights_path)
        if self.segformer_wrapper.is_loaded:
            self.segformer_weights_path = weights_path
            self.use_mock_segmentation = False
            print(f"[Pipeline] SegFormer đã kích hoạt từ: {weights_path}")
            return True
        else:
            self.segformer_weights_path = None
            self.use_mock_segmentation = True
            print("[Pipeline] Hoạt động ở chế độ Segmentation Synthetic Mock.")
            return False

    def get_model_status(self):
        """Trả về tình trạng kết nối hiện tại của cả 2 mô hình"""
        yolo_live = (self.yolo_model is not None) and (not self.use_mock_detection)
        seg_live = (self.segformer_wrapper is not None) and self.segformer_wrapper.is_loaded
        seg_backbone = getattr(self.segformer_wrapper, "backbone_name", "SegFormer") if seg_live else "SegFormer"

        return {
            "yolo": {
                "loaded": yolo_live,
                "weights": self.yolo_weights_path if yolo_live else "Chưa có trọng số",
                "mode": "Live (YOLOv12 + ByteTrack)" if yolo_live else "Synthetic Mock (Giả lập BBox)"
            },
            "segformer": {
                "loaded": seg_live,
                "backbone": seg_backbone,
                "weights": self.segformer_weights_path if seg_live else "Chưa có trọng số",
                "mode": f"Live ({seg_backbone})" if seg_live else "Synthetic Mock (Mặt nạ hình học)"
            }
        }

    def reset(self):
        """Đặt lại trạng thái của pipeline khi bắt đầu video mới"""
        self.engine = TrafficContextEngine(fps=self.fps)
        self.last_masks = None
        self.last_detections = None
        self.ema_fps = float(self.fps)
        self._buffer_shape = None
        if self.yolo_model is not None and getattr(self.yolo_model, "predictor", None) is not None:
            for trk in getattr(self.yolo_model.predictor, "trackers", []):
                if hasattr(trk, "reset"):
                    trk.reset()
        if self.segformer_wrapper and hasattr(self.segformer_wrapper, "reset"):
            self.segformer_wrapper.reset()

    def _detect_and_track(self, frame, frame_idx, det_interval=2):
        """Chạy YOLO + ByteTrack với Interleaved Caching (lịch trình đan xen không trùng nhịp SegFormer)"""
        # Tái sử dụng kết quả nhận diện ở frame lẻ nếu bật det_interval > 1
        if self.last_detections is not None and det_interval > 1 and (frame_idx % det_interval != 0):
            return self.last_detections

        detections = []

        if self.yolo_model and not self.use_mock_detection:
            try:
                h_orig, w_orig = frame.shape[:2]
                
                # Tối ưu hóa cho mọi video 4K/FullHD: Lưới 640x384 (16:9 chuẩn chia hết cho 32, bắt trọn cả xe ở xa)
                f_det = cv2.resize(frame, (640, 384), interpolation=cv2.INTER_LINEAR)
                scale_x = w_orig / 640.0
                scale_y = h_orig / 384.0

                results = self.yolo_model.track(
                    f_det,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.38,
                    imgsz=(384, 640),
                    verbose=False,
                    device=self.device_target
                )[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.int().cpu().numpy()
                    names = results.names
                    track_ids = (
                        results.boxes.id.int().cpu().numpy()
                        if results.boxes.id is not None
                        else np.arange(len(boxes))
                    )

                    for box, track_id, cls_id in zip(boxes, track_ids, classes):
                        raw_class_name = str(names.get(cls_id, cls_id))
                        raw_lower = raw_class_name.lower()
                        
                        # Chuẩn hóa tên lớp về đúng chuẩn đề tài: Vehicle, Pedestrian, Traffic Light
                        mapped_class = "Vehicle"
                        if any(k in raw_lower for k in ["pedestrian", "person", "rider", "walker", "bicycle", "bike"]):
                            mapped_class = "Pedestrian"
                        elif any(k in raw_lower for k in ["traffic light", "traffic_light", "light", "signal"]):
                            mapped_class = "Traffic Light"
                        elif any(k in raw_lower for k in ["car", "bus", "truck", "van", "motorcycle", "vehicle"]):
                            mapped_class = "Vehicle"
                        elif cls_id == 0:
                            mapped_class = "Vehicle"
                        elif cls_id == 1:
                            mapped_class = "Pedestrian"
                        elif cls_id == 2:
                            mapped_class = "Traffic Light"

                        # Khôi phục tọa độ chính xác về độ phân giải gốc của video
                        bx1 = int(box[0] * scale_x)
                        by1 = int(box[1] * scale_y)
                        bx2 = int(box[2] * scale_x)
                        by2 = int(box[3] * scale_y)

                        # Lọc nhiễu hình học không gian (Plausibility Filter) cho đèn giao thông
                        bw = bx2 - bx1
                        bh = by2 - by1
                        if mapped_class == "Traffic Light":
                            aspect = bw / max(1, bh)
                            # Đèn giao thông dạng cột đứng có w/h <= 1.25, không quá to (<= 15% chiều cao ảnh)
                            if aspect > 1.25 or (bh > h_orig * 0.15) or (bw > w_orig * 0.10):
                                continue
                            if by1 > h_orig * 0.70 or by2 < h_orig * 0.05:
                                continue

                        detections.append({
                            "id": int(track_id),
                            "class": mapped_class,
                            "bbox": [bx1, by1, bx2, by2]
                        })
            except Exception:
                detections = []

        self.last_detections = [dict(d) for d in detections]
        return detections

    def _segment_scene(self, frame, frame_idx=0, seg_interval=4):
        """Chạy SegFormer đan xen ở nhịp lẻ (frame_idx % seg_interval == 1) để không đụng độ YOLO ở nhịp chẵn"""
        h, w = frame.shape[:2]
        if self.segformer_wrapper and self.segformer_wrapper.is_loaded:
            # Đan xen: Chạy SegFormer ở frame 0 hoặc các frame lẻ (1, 5, 9...) để triệt tiêu độ trễ đỉnh (frame spike)
            if self.last_masks is None or (seg_interval <= 1) or (frame_idx % seg_interval == 1):
                self.last_masks = self.segformer_wrapper.predict(frame)
            return self.last_masks
        empty = np.zeros((h, w), dtype=np.uint8)
        return {"drivable": empty, "lane": empty, "lane_core": empty, "raw_pred": None}

    def draw_hud_visuals(self, frame, detections, masks, context_log, 
                         show_masks=True, show_lanes=True, show_boxes=True, 
                         show_graph=True, show_hud=True):
        """
        Ủy quyền trực quan hóa cho ADASVisualizer (Clean Architecture - View Layer).
        """
        return self.visualizer.draw_hud_visuals(
            frame=frame,
            detections=detections,
            masks=masks,
            context_log=context_log,
            show_masks=show_masks,
            show_lanes=show_lanes,
            show_boxes=show_boxes,
            show_graph=show_graph,
            show_hud=show_hud
        )

    def run_inference(self, frame, frame_idx=0, weather="Nang rao", 
                      show_masks=True, show_lanes=True, show_boxes=True, 
                      show_graph=True, show_hud=True,
                      seg_interval=6, det_interval=3):
        """
        Thực hiện toàn bộ luồng nhận thức trên 1 frame:
        Frame -> Detection & Tracking -> Segmentation -> Context Engine -> HUD Visualized Frame + JSON Log
        
        Args:
            frame (np.ndarray): Khung hình ảnh gốc (BGR từ OpenCV)
            frame_idx (int): Thứ tự frame trong video
            weather (str): Điều kiện thời tiết
            seg_interval (int): Khoảng cách frame cập nhật SegFormer (Mask Caching)
            det_interval (int): Khoảng cách frame cập nhật YOLO (Interleaved Scheduling)
            
        Returns:
            tuple: (annotated_frame, context_log)
        """
        start_time = time.time()
        h, w = frame.shape[:2]

        # 1. Phát hiện & Bám vết đối tượng (YOLO + ByteTrack ở nhịp chẵn)
        detections = self._detect_and_track(frame, frame_idx, det_interval=det_interval)

        # 2. Phân đoạn ngữ nghĩa làn đường & vùng di chuyển (SegFormer ở nhịp lẻ)
        masks = self._segment_scene(frame, frame_idx=frame_idx, seg_interval=seg_interval)

        # 3. Suy luận ngữ cảnh giao thông (Traffic Context Representation Engine)
        context_log = self.engine.process_frame(
            detections=detections,
            masks=masks,
            weather_label=weather,
            frame_idx=frame_idx,
            frame_shape=(h, w),
            raw_frame=frame
        )

        # 4. Trực quan hóa hình ảnh giao diện ADAS
        annotated_frame = self.draw_hud_visuals(
            frame=frame,
            detections=detections,
            masks=masks,
            context_log=context_log,
            show_masks=show_masks,
            show_lanes=show_lanes,
            show_boxes=show_boxes,
            show_graph=show_graph,
            show_hud=show_hud
        )

        elapsed = time.time() - start_time
        fps_realtime = 1.0 / elapsed if elapsed > 0 else self.fps
        self.ema_fps = 0.85 * self.ema_fps + 0.15 * min(fps_realtime, 120.0)
        context_log["summary"]["fps"] = round(self.ema_fps, 1)

        return annotated_frame, context_log
