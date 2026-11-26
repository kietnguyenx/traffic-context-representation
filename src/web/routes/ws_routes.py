"""
WebSocket Route cho Luồng Phân Tích Thời Gian Thực (Real-Time Binary + JSON Stream)
Hỗ trợ 2 chiều:
- Server -> Client: Binary JPEG frames siêu tốc + JSON Telemetry / Context Log / Conclusion
- Client -> Server: Tín hiệu bật/tắt 4 công tắc AI trực tiếp (type="toggles") hoặc hủy (type="stop")
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..state import app_state
from ..video_processor import stream_video_analysis

router = APIRouter(tags=["ADAS WebSocket"])


@router.websocket("/ws/analyze")
async def ws_analyze_video(websocket: WebSocket):
    await websocket.accept()

    async def _listen_client_commands():
        """Lắng nghe lệnh điều khiển Live Toggles / Stop từ trình duyệt trong lúc đang stream"""
        try:
            while True:
                msg = await websocket.receive_json()
                mtype = msg.get("type")
                if mtype == "toggles":
                    app_state.update_toggles(msg.get("data", {}))
                elif mtype == "stop":
                    app_state.cancel_event.set()
        except (WebSocketDisconnect, Exception):
            app_state.cancel_event.set()

    listener_task = asyncio.create_task(_listen_client_commands())
    try:
        await stream_video_analysis(
            state=app_state,
            send_bytes=websocket.send_bytes,
            send_json=websocket.send_json,
        )
    except WebSocketDisconnect:
        app_state.cancel_event.set()
    finally:
        listener_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
