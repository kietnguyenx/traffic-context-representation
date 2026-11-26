"""
Điểm Khởi Chạy Web Server HTML/CSS/JS + FastAPI (High-FPS Real-Time ADAS Dashboard)
Chạy bằng lệnh:
    python src/web_server.py
Sau đó mở trình duyệt: http://localhost:8501
"""

import os
import sys
import uvicorn

# Đảm bảo thư mục src nằm trong sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from web.app_factory import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8501"))
    print("=" * 68)
    print("🚗 TRAFFIC CONTEXT REPRESENTATION — HTML/CSS/JS + FastAPI SERVER")
    print(f"🌐 Mở trình duyệt tại: http://localhost:{port}")
    print("=" * 68)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    )
