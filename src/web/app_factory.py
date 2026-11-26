"""
FastAPI Application Factory & Lifespan Management
Tự động nạp sẵn 2 mô hình YOLOv12-L và SegFormer MiT-B2 lên GPU khi khởi động server.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Đảm bảo import được DrivingPerceptionPipeline từ thư mục src/
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import DrivingPerceptionPipeline
from .state import app_state
from .routes.api_routes import router as api_router
from .routes.ws_routes import router as ws_router

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Nạp mô hình AI một lần duy nhất lúc khởi động server (Zero Cold-Start cho trình duyệt)"""
    if app_state.pipeline is None:
        app_state.set_pipeline(DrivingPerceptionPipeline(fps=30))
    yield
    app_state.clear_video()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Traffic Context Representation | Real-Time ADAS",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.include_router(api_router)
    app.include_router(ws_router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon():
        return FileResponse(
            os.path.join(STATIC_DIR, "favicon.svg"),
            media_type="image/svg+xml",
        )

    return app
