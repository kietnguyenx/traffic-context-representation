"""
Module Quản Lý Trạng Thái Phiên Làm Việc & Cấu Hình Live Toggles (Thread-Safe State Store)
Tách rời quản lý tài nguyên file tạm, trạng thái pipeline và cờ điều khiển lớp AI thời gian thực.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DisplayToggles:
    """4 công tắc bật/tắt lớp hiển thị AI (có thể thay đổi trực tiếp ngay khi video đang chạy)"""
    show_boxes: bool = True
    show_drivable: bool = True
    show_lanes: bool = True
    show_graph: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return {
            "show_boxes": self.show_boxes,
            "show_drivable": self.show_drivable,
            "show_lanes": self.show_lanes,
            "show_graph": self.show_graph,
        }


@dataclass
class VideoMetadata:
    """Thông số kỹ thuật của video hành trình đã nạp"""
    file_name: str
    file_size: int
    width: int
    height: int
    fps: float
    total_frames: int
    poster_b64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_size": self.file_size,
            "resolution": f"{self.width}x{self.height}",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "poster_b64": self.poster_b64,
        }


class AppStateManager:
    """
    Quản lý tập trung trạng thái toàn hệ thống cho Web Server (Singleton Thread-Safe):
    - Lưu trữ instance DrivingPerceptionPipeline duy nhất trên GPU
    - Quản lý file video gốc & file video H.264 sau phân tích (tự dọn file tạm)
    - Quản lý sự kiện hủy luồng (cancel_event) khi người dùng bấm Dừng hoặc Xóa video
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.pipeline = None
        self.toggles = DisplayToggles()
        self.uploaded_path: Optional[str] = None
        self.analyzed_path: Optional[str] = None
        self.video_meta: Optional[VideoMetadata] = None
        self.conclusion_payload: Optional[Dict[str, Any]] = None
        self.is_analyzing: bool = False
        self.cancel_event = threading.Event()
        self.analysis_lock = threading.Lock()

    def set_pipeline(self, pipeline) -> None:
        with self._lock:
            self.pipeline = pipeline

    def update_toggles(self, data: Dict[str, Any]) -> Dict[str, bool]:
        with self._lock:
            for key in ("show_boxes", "show_drivable", "show_lanes", "show_graph"):
                if key in data:
                    setattr(self.toggles, key, bool(data[key]))
            return self.toggles.to_dict()

    def get_toggles(self) -> DisplayToggles:
        with self._lock:
            return DisplayToggles(
                show_boxes=self.toggles.show_boxes,
                show_drivable=self.toggles.show_drivable,
                show_lanes=self.toggles.show_lanes,
                show_graph=self.toggles.show_graph,
            )

    def _safe_remove(self, path: Optional[str]) -> None:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def set_uploaded_video(self, temp_path: str, meta: VideoMetadata) -> None:
        with self._lock:
            self.cancel_event.set()
            self._safe_remove(self.uploaded_path)
            self._safe_remove(self.analyzed_path)
            self.uploaded_path = temp_path
            self.analyzed_path = None
            self.video_meta = meta
            self.conclusion_payload = None
            if self.pipeline is not None:
                self.pipeline.reset()

    def set_analyzed_video(self, out_path: str, conclusion_payload: Dict[str, Any]) -> None:
        with self._lock:
            if self.analyzed_path and self.analyzed_path != out_path:
                self._safe_remove(self.analyzed_path)
            self.analyzed_path = out_path
            self.conclusion_payload = conclusion_payload

    def clear_video(self) -> None:
        with self._lock:
            self.cancel_event.set()
            self._safe_remove(self.uploaded_path)
            self._safe_remove(self.analyzed_path)
            self.uploaded_path = None
            self.analyzed_path = None
            self.video_meta = None
            self.conclusion_payload = None
            if self.pipeline is not None:
                self.pipeline.reset()

    def get_status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            model_status = self.pipeline.get_model_status() if self.pipeline else {
                "yolo": {"loaded": False},
                "segformer": {"loaded": False, "backbone": "MiT-B2"},
            }
            return {
                "models": model_status,
                "has_cuda": bool(getattr(self.pipeline, "has_cuda", False)) if self.pipeline else False,
                "toggles": self.toggles.to_dict(),
                "has_video": self.uploaded_path is not None and os.path.exists(self.uploaded_path),
                "has_analyzed_video": self.analyzed_path is not None and os.path.exists(self.analyzed_path),
                "is_analyzing": self.is_analyzing,
                "video_meta": self.video_meta.to_dict() if self.video_meta else None,
                "conclusion": self.conclusion_payload,
            }


# Global thread-safe state singleton
app_state = AppStateManager()
