"""
REST API Routes (FastAPI Router)
Cung cấp các endpoint quản lý trạng thái mô hình, nạp/xóa video, phát lại video đã phân tích và cập nhật Live Toggles.
"""

import os
import shutil
import base64
import tempfile
from typing import Dict, Any

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..state import app_state, VideoMetadata

router = APIRouter(prefix="/api", tags=["ADAS API"])


class TogglesPayload(BaseModel):
    show_boxes: bool = True
    show_drivable: bool = True
    show_lanes: bool = True
    show_graph: bool = True


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Trả về trạng thái 2 mô hình AI (YOLOv12-L & SegFormer MiT-B2) và video hiện tại"""
    return app_state.get_status_snapshot()


@router.post("/toggles")
async def update_display_toggles(payload: TogglesPayload) -> Dict[str, Any]:
    """Cập nhật tức thời 4 nút gạt lớp hiển thị AI (tác dụng ngay kể cả khi video đang chạy)"""
    updated = app_state.update_toggles(payload.model_dump())
    return {"ok": True, "toggles": updated}


@router.post("/video/upload")
async def upload_dashcam_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Nhận file video hành trình (.mp4, .avi, .mov), trích xuất metadata & ảnh poster đầu tiên tức thì"""
    filename = file.filename or "dashcam_video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(status_code=400, detail="Only MP4, AVI, MOV, and MKV video formats are supported.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".mp4")
    try:
        shutil.copyfileobj(file.file, tfile, length=16 * 1024 * 1024)
        tfile.flush()
    finally:
        tfile.close()
        await file.close()

    file_size = os.path.getsize(tfile.name)

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        try:
            os.remove(tfile.name)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Failed to open the uploaded video file.")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = round(cap.get(cv2.CAP_PROP_FPS) or 24.0, 1)
    in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ret_p, first_f = cap.read()
    cap.release()

    poster_b64 = None
    if ret_p and first_f is not None:
        if first_f.shape[1] > 960:
            p_scale = 960.0 / first_f.shape[1]
            first_f = cv2.resize(
                first_f,
                (960, max(2, int(first_f.shape[0] * p_scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok_p, p_buf = cv2.imencode(".jpg", first_f, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        if ok_p:
            poster_b64 = base64.b64encode(p_buf.tobytes()).decode("ascii")

    meta = VideoMetadata(
        file_name=filename,
        file_size=file_size,
        width=in_w,
        height=in_h,
        fps=in_fps,
        total_frames=in_frames,
        poster_b64=poster_b64,
    )
    app_state.set_uploaded_video(tfile.name, meta)
    return {"ok": True, "video_meta": meta.to_dict()}


@router.post("/video/clear")
async def clear_uploaded_video() -> Dict[str, Any]:
    """Xóa sạch video đã nạp, hủy luồng đang chạy và giải phóng bộ đệm tức thì (0ms)"""
    app_state.clear_video()
    return {"ok": True}


@router.post("/analysis/stop")
async def stop_analysis() -> Dict[str, Any]:
    """Dừng luồng phân tích hiện tại an toàn"""
    app_state.cancel_event.set()
    return {"ok": True}


@router.get("/video/analyzed")
async def get_analyzed_video():
    """Trả về file video MP4 H.264 sau phân tích hỗ trợ Range Requests để tua mượt trên <video>"""
    path = app_state.analyzed_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No analyzed video available.")
    return FileResponse(path, media_type="video/mp4", filename="adas_analyzed.mp4")
