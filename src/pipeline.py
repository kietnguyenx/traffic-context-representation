import cv2
import numpy as np
from context_engine import TrafficContextEngine

class DrivingPerceptionPipeline:
    def __init__(self, yolo_weights=None, segformer_weights=None, fps=30):
        self.fps = fps
        self.engine = TrafficContextEngine(fps=fps)
        self.yolo_model = None
        self.segformer_model = None
        
        # Nạp mô hình của Thành viên 1 khi có file yolo_best.pt
        if yolo_weights:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_weights)
            except ImportError:
                print("Chưa cài ultralytics hoặc chưa có weights YOLO.")
        
        # Nạp mô hình của Thành viên 2 khi có file segformer_best.pth
        if segformer_weights:
            # Sẽ kết nối class loader của Thành viên 2
            pass

    def run_inference(self, frame, frame_idx=0, weather="Nang rao"):
        """
        Xử lý 1 frame qua toàn bộ luồng:
        Frame -> YOLO (BBoxes + ID) & SegFormer (Masks) -> Context Engine -> Visualized Frame + JSON Log
        """
        h, w = frame.shape[:2]
        detections = []
        masks = {}

        # 1. Chạy YOLOv8 + ByteTrack (nếu đã nạp model của TV1)
        if self.yolo_model:
            results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
            if results.boxes and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                classes = results.boxes.cls.int().cpu().numpy()
                names = results.names

                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    detections.append({
                        "id": int(track_id),
                        "class": names[cls_id],
                        "bbox": [int(x) for x in box]
                    })
        else:
            # Dữ liệu giả lập chạy thử nghiệm khi chưa có weights
            detections = [
                {"id": 1, "class": "Vehicle", "bbox": [int(w*0.4), int(h*0.5), int(w*0.6), int(h*0.8)]}
            ]

        # 2. Suy luận ngữ cảnh từ Context Engine của bạn
        context_log = self.engine.process_frame(
            detections=detections,
            masks=masks,
            weather_label=weather,
            frame_idx=frame_idx,
            frame_shape=(h, w)
        )

        # 3. Vẽ đè Bounding Box và cảnh báo trực quan
        annotated_frame = frame.copy()
        for obj in detections:
            x1, y1, x2, y2 = obj['bbox']
            label = f"ID:{obj['id']} {obj['class']}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        y_offset = 30
        for warn in context_log["warnings"]:
            cv2.putText(annotated_frame, warn, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25

        return annotated_frame, context_log