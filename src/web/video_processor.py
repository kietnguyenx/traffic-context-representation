"""
Module Xử Lý Video Bất Đồng Bộ & Đẩy Luồng Thời Gian Thực (Async Video Processor)
Kiến trúc 3 luồng Producer - GPU Worker - Async Writer cho FPS tối đa:
- Reader Thread: Đọc & resize trước khung hình qHD 960x540 (zero I/O wait)
- GPU Inference Worker: Chạy YOLOv12-L + SegFormer-B2 + ContextEngine + ADASVisualizer
- Writer Thread: Ghi nén H.264 MP4 chạy song song để phát lại mượt mà ngay sau khi phân tích xong
- Binary JPEG WebSocket Stream: Truyền thẳng byte ảnh xuống trình duyệt (không tốn chi phí Base64)
"""

import os
import time
import queue
import tempfile
import threading
import asyncio
from typing import Callable, Awaitable, Dict, Any

import cv2
import numpy as np

from .state import AppStateManager
from .telemetry_service import TripTelemetryAccumulator, format_context_log_en


def _open_h264_writer(out_path: str, fps: float, size: tuple) -> cv2.VideoWriter:
    """Mở VideoWriter ưu tiên codec H.264 tương thích trực tiếp thẻ <video> trên Chrome/Edge"""
    writer = cv2.VideoWriter(
        out_path,
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        float(fps),
        size,
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            float(fps),
            size,
        )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            size,
        )
    try:
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 90)
    except Exception:
        pass
    return writer


async def stream_video_analysis(
    state: AppStateManager,
    send_bytes: Callable[[bytes], Awaitable[None]],
    send_json: Callable[[Dict[str, Any]], Awaitable[None]],
) -> None:
    """
    Thực thi phân tích video toàn trình và đẩy đồng thời:
    - Binary JPEG frames (zero base64 overhead, tối đa FPS hiển thị)
    - JSON Telemetry & Context Log định kỳ
    - Gói Tổng Kết Hành Trình + URL phát lại video MP4 H.264 khi hoàn tất
    """
    if not state.uploaded_path or not os.path.exists(state.uploaded_path):
        await send_json({"type": "error", "message": "No dashcam video loaded."})
        return

    pipeline = state.pipeline
    if pipeline is None:
        await send_json({"type": "error", "message": "AI perception pipeline is not ready."})
        return

    loop = asyncio.get_running_loop()
    state.cancel_event.clear()
    state.is_analyzing = True

    input_video_path = state.uploaded_path
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    if not fps_source or fps_source <= 0 or np.isnan(fps_source):
        fps_source = 24.0

    if total_frames <= 0:
        cap.release()
        state.is_analyzing = False
        await send_json({"type": "error", "message": "Unable to decode frames from the uploaded video."})
        return

    # Tự động chuẩn hóa kích thước qHD 960x540 (16:9) cho tốc độ cao và nét căng
    if orig_w > 960 or orig_h > 540:
        scale = min(960.0 / orig_w, 540.0 / orig_h)
        target_w = int(round(orig_w * scale / 2.0) * 2)
        target_h = int(round(orig_h * scale / 2.0) * 2)
    else:
        target_w = max(2, (orig_w // 2) * 2)
        target_h = max(2, (orig_h // 2) * 2)

    frame_stride = 2 if fps_source >= 45.0 else 1
    out_fps = fps_source / frame_stride
    effective_total = max(1, (total_frames + frame_stride - 1) // frame_stride)

    has_cuda = bool(getattr(pipeline, "has_cuda", False))
    det_step = 3 if has_cuda else 4
    seg_step = 6 if has_cuda else 8

    out_processed_path = os.path.join(
        tempfile.gettempdir(), f"adas_web_analyzed_{int(time.time() * 1000)}.mp4"
    )
    video_writer = _open_h264_writer(out_processed_path, out_fps, (target_w, target_h))

    raw_queue: queue.Queue = queue.Queue(maxsize=24)
    writer_queue: queue.Queue = queue.Queue(maxsize=32)
    out_stream_queue: asyncio.Queue = asyncio.Queue(maxsize=6)

    def _enqueue_async(item: Any) -> None:
        """Đẩy gói dữ liệu từ worker thread sang asyncio loop an toàn (drop frame cũ nếu client chậm)"""
        def _put():
            if item[0] == "frame" and out_stream_queue.full():
                try:
                    out_stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                out_stream_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
        loop.call_soon_threadsafe(_put)

    def _reader_thread():
        raw_idx = 0
        try:
            while capture_is_open := cap.isOpened():
                if state.cancel_event.is_set():
                    break
                if frame_stride > 1 and (raw_idx % frame_stride != 0):
                    if not cap.grab():
                        break
                    raw_idx += 1
                    continue
                ret_r, raw_f = cap.read()
                if not ret_r:
                    break
                raw_idx += 1
                if raw_f.shape[1] != target_w or raw_f.shape[0] != target_h:
                    raw_f = cv2.resize(raw_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                while not state.cancel_event.is_set():
                    try:
                        raw_queue.put(raw_f, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            cap.release()
            try:
                raw_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

    def _writer_thread():
        try:
            while True:
                item_f = writer_queue.get()
                if item_f is None:
                    break
                if video_writer.isOpened():
                    video_writer.write(item_f)
        finally:
            if video_writer.isOpened():
                video_writer.release()

    def _inference_worker():
        accumulator = TripTelemetryAccumulator(target_h=target_h, out_fps=out_fps)
        idx = 0
        last_frame_sent_t = 0.0
        selected_weather = "Tự động (AI Optical)"

        with state.analysis_lock:
            pipeline.reset()
            pipeline.fps = out_fps
            pipeline.engine.fps = out_fps

            try:
                while not state.cancel_event.is_set():
                    try:
                        small_f = raw_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if small_f is None:
                        break

                    # Đọc trực tiếp trạng thái 4 công tắc LIVE (thay đổi ngay giữa lúc đang stream!)
                    toggles = state.get_toggles()

                    ann_f, log = pipeline.run_inference(
                        frame=small_f,
                        frame_idx=idx,
                        weather=selected_weather,
                        show_masks=toggles.show_drivable,
                        show_lanes=toggles.show_lanes,
                        show_boxes=toggles.show_boxes,
                        show_graph=toggles.show_graph,
                        seg_interval=seg_step,
                        det_interval=det_step,
                    )

                    if not state.cancel_event.is_set():
                        writer_queue.put(ann_f)

                    telemetry_pkt = accumulator.record_frame(log)

                    # Đẩy binary JPEG (nhanh gấp 3x base64, tối đa ~30 FPS mượt mà)
                    now_t = time.time()
                    if (now_t - last_frame_sent_t >= 0.032) or (idx + 1 >= effective_total):
                        ok_enc, jpeg_buf = cv2.imencode(
                            ".jpg", ann_f, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        )
                        if ok_enc:
                            _enqueue_async(("frame", jpeg_buf.tobytes()))
                            last_frame_sent_t = now_t

                    # Đẩy Telemetry & Progress mỗi 6 frames (~5 lần/giây mượt mà) và JSON Log mỗi 12 frames
                    if idx % 6 == 0 or (idx + 1 >= effective_total):
                        payload: Dict[str, Any] = {
                            "type": "telemetry",
                            "current_frame": idx + 1,
                            "total_frames": effective_total,
                            "out_fps": out_fps,
                            "telemetry": telemetry_pkt,
                        }
                        if idx % 12 == 0 or (idx + 1 >= effective_total):
                            payload["context_log"] = format_context_log_en(log)
                        _enqueue_async(("json", payload))

                    idx += 1
            finally:
                writer_queue.put(None)

        if state.cancel_event.is_set():
            _enqueue_async(("done", {"type": "cancelled"}))
            return

        writer_t.join(timeout=15.0)
        conclusion_payload = accumulator.build_conclusion(analyzed_count=idx)
        state.set_analyzed_video(out_processed_path, conclusion_payload)
        _enqueue_async(
            (
                "done",
                {
                    "type": "completed",
                    "video_url": f"/api/video/analyzed?t={int(time.time() * 1000)}",
                    "conclusion": conclusion_payload,
                },
            )
        )

    reader_t = threading.Thread(target=_reader_thread, daemon=True)
    writer_t = threading.Thread(target=_writer_thread, daemon=True)
    worker_t = threading.Thread(target=_inference_worker, daemon=True)
    reader_t.start()
    writer_t.start()
    worker_t.start()

    try:
        while True:
            kind, data = await out_stream_queue.get()
            if kind == "frame":
                await send_bytes(data)
            elif kind == "json":
                await send_json(data)
            elif kind == "done":
                await send_json(data)
                break
    except Exception:
        state.cancel_event.set()
    finally:
        state.is_analyzing = False
