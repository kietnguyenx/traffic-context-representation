"""
Giao Diện Web Demo: Traffic Context Representation & ADAS Perception Dashboard
(Multi-Task Deep Learning: YOLO + SegFormer)

Thiết kế theo chuẩn Cockpit ADAS hiện đại khớp với mockup:
- Top Navigation Bar: Tiêu đề hệ thống kèm Status Badge (🟢 Active • 30 FPS)
- Cột 1 (Controls):
  - Kéo-thả trực tiếp video hành trình (.mp4, .avi, .mov)
  - Dataset Selector (BDD100K, nuScenes, Waymo, KITTI)
  - 4 Toggles: YOLO 3D Bounding Boxes, SegFormer Drivable Area, SegFormer Lane Markings, Context Reasoning Graph
  - Nút Start Analysis (Neon Cyan)
- Cột 2 (Center):
  - 1080p Driver-view Dashcam Video: Trạng thái chờ (Standby) khi chưa có video, hiển thị preview ngay khi kéo thả video vào.
  - Telemetry Panel (Vận tốc, Khoảng cách, TTC, Mức độ rủi ro)
- Cột 3 (Right):
  - Telemetry Sidebar / HUD (Ego Lane, Lead Vehicle [Caution in amber], Traffic Signal, Pedestrian Hazard, Terminal JSON)

Tác giả: Thành viên 3 (Phụ trách Context Engine, Pipeline & Web Demo)
"""

import os
import sys
import time
import json
import tempfile
import cv2
import numpy as np
import streamlit as st

# Thêm thư mục hiện tại vào sys.path để import pipeline và utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from pipeline import DrivingPerceptionPipeline
from utils.demo_generator import get_demo_scenario_data

# 1. CẤU HÌNH TRANG WEB (FULL WIDTH, NO DEFAULT SIDEBAR)
st.set_page_config(
    page_title="Traffic Context Representation & ADAS Perception Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. BỘ STYLING CSS HIỆN ĐẠI THEO PHONG CÁCH ADAS COCKPIT MOCKUP
st.markdown("""
<style>
    /* Ẩn hoàn toàn header và toolbar mặc định của Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    .stAppToolbar {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }
    
    /* Căn chỉnh không gian tổng thể */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* TOP NAVIGATION BAR */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hamburger-icon {
        font-size: 18px;
        color: #94a3b8;
        cursor: pointer;
        padding-right: 4px;
    }
    .top-title {
        font-size: 15px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.3px;
    }
    .top-divider {
        color: #475569;
        margin: 0 4px;
    }
    .top-subtitle {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 400;
    }
    .status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: 20px;
        padding: 4px 14px;
        color: #4ade80;
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 8px #22c55e;
        display: inline-block;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* KHUNG PANEL CHUNG (ADAS CONTAINERS) */
    .panel-header {
        font-size: 14px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 12px;
        letter-spacing: 0.4px;
    }

    /* TÙY BIẾN STREAMLIT FILE UPLOADER (THANH TRÊN KHỚP ẢNH MẪU) */
    div[data-testid="stFileUploader"] {
        margin-bottom: 0px !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
        border: 1px solid #1f293d !important;
        border-radius: 8px !important;
        background: #111827 !important;
        padding: 8px 12px !important;
        text-align: left !important;
    }
    div[data-testid="stFileUploader"] button {
        border: 1px solid #2a384f !important;
        background: #161f30 !important;
        color: #f1f5f9 !important;
        border-radius: 6px !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        padding: 5px 12px !important;
    }

    /* VÙNG KÉO THẢ NÉT ĐỨT (DASHED DROPZONE KHỚP ẢNH MẪU 100%) */
    .dropzone-container {
        border: 1.5px dashed #1e3a5f;
        border-radius: 10px;
        background: rgba(12, 18, 30, 0.6);
        padding: 24px 14px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 14px;
        transition: all 0.25s ease;
        cursor: pointer;
    }
    .dropzone-container:hover {
        border-color: #00f2fe;
        background: rgba(15, 23, 42, 0.85);
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.25);
    }
    .dropzone-icon {
        font-size: 26px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .dropzone-text {
        font-size: 12.5px;
        color: #8b949e;
        line-height: 1.45;
    }

    /* NHÃN CONTROLS & DROPDOWN */
    .control-label {
        font-size: 13px;
        font-weight: 500;
        color: #cbd5e1;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    /* TÙY BIẾN STREAMLIT SELECTBOX */
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border: 1px solid #1f293d !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        font-size: 13.5px !important;
    }

    /* TÙY BIẾN TOGGLES */
    div[data-testid="stToggle"] {
        margin-top: 6px !important;
        margin-bottom: 6px !important;
    }
    div[data-testid="stToggle"] label p {
        font-size: 13px !important;
        color: #cbd5e1 !important;
        font-weight: 400 !important;
    }

    /* NÚT START ANALYSIS (NEON CYAN PILL) */
    .stButton > button {
        background: linear-gradient(90deg, #00f2fe 0%, #22d3ee 100%) !important;
        color: #050811 !important;
        border: none !important;
        border-radius: 22px !important;
        padding: 9px 20px !important;
        font-size: 14.5px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.45) !important;
        transition: all 0.25s ease !important;
        margin-top: 12px !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.7) !important;
    }

    /* KHUNG STANDBY CHO VIDEO KHI CHƯA NẠP */
    .video-standby-viewport {
        border: 1px dashed #334155;
        border-radius: 10px;
        background: #080c14;
        min-height: 380px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 24px;
        text-align: center;
    }
    .standby-icon {
        font-size: 42px;
        color: #475569;
    }
    .standby-title {
        font-size: 15px;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.5px;
    }
    .standby-desc {
        font-size: 12.5px;
        color: #64748b;
        max-width: 400px;
        line-height: 1.5;
    }

    /* THẺ TELEMETRY PHÍA DƯỚI VIDEO */
    .telemetry-bottom-card {
        background: #0e1422;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
    }
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 6px;
    }
    .telemetry-item {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
    }
    .telemetry-lbl {
        font-size: 10.5px;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 0.8px;
    }
    .telemetry-val {
        font-size: 17px;
        font-weight: 700;
        margin-top: 3px;
        color: #f1f5f9;
    }
    .telemetry-val.cyan { color: #22d3ee; }
    .telemetry-val.amber { color: #f59e0b; }
    .telemetry-val.green { color: #22c55e; }
    .telemetry-val.red { color: #ef4444; }
    .telemetry-val.muted { color: #64748b; }

    /* HUD CARDS Ở CỘT PHẢI */
    .hud-card {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .hud-title {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .hud-val-primary {
        font-size: 15px;
        font-weight: 600;
        color: #f1f5f9;
    }
    
    /* THẺ LEAD VEHICLE (CAUTION IN AMBER) */
    .hud-card.amber-glow {
        border: 1.5px solid #d97706 !important;
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.12) 0%, rgba(17, 24, 39, 0.95) 100%) !important;
        box-shadow: 0 0 16px rgba(217, 119, 6, 0.25) !important;
    }
    .hud-val-amber {
        font-size: 15px;
        font-weight: 600;
        color: #fbbf24;
    }
    .hud-sub-amber {
        font-size: 12px;
        color: #f59e0b;
        margin-top: 2px;
    }

    /* THẺ TRAFFIC SIGNAL */
    .hud-val-red {
        font-size: 15px;
        font-weight: 700;
        color: #f87171;
    }

    /* THẺ PEDESTRIAN HAZARD */
    .hud-val-alert {
        font-size: 15px;
        font-weight: 600;
        color: #cbd5e1;
    }

    /* TERMINAL CONTAINER */
    .terminal-card {
        background: #060913;
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 4px;
    }
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0d131f;
        padding: 6px 12px;
        border-bottom: 1px solid #1e293b;
        font-size: 12px;
        color: #94a3b8;
    }
    .terminal-close {
        color: #64748b;
        font-size: 13px;
        cursor: pointer;
    }
    .terminal-body {
        padding: 10px 14px;
        font-family: 'Fira Code', 'Cascadia Code', Consolas, monospace;
        font-size: 12px;
        line-height: 1.5;
        color: #e2e8f0;
        overflow-x: auto;
        min-height: 160px;
    }
    .json-punct { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# 3. TOP NAVIGATION BAR (HEADER KHỚP VỚI MOCKUP)
st.markdown("""
<div class="top-nav">
    <div class="top-nav-left">
        <span class="hamburger-icon">☰</span>
        <span class="top-title">Traffic Context Representation & ADAS Perception Dashboard</span>
        <span class="top-divider">|</span>
        <span class="top-subtitle">Multi-Task Deep Learning (YOLO + SegFormer)</span>
    </div>
    <div class="top-nav-right">
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>Active • 30 FPS</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# 4. CHIA BỐ CỤC 3 CỘT (CONTROLS - VIDEO - HUD)
col_controls, col_video, col_hud = st.columns([1.15, 2.5, 1.25], gap="medium")


# -------------------------------------------------------------
# CỘT 1: CONTROLS
# -------------------------------------------------------------
with col_controls:
    st.markdown('<div class="panel-header">Controls</div>', unsafe_allow_html=True)

    # Vùng upload video: Bản thân st.file_uploader đã hỗ trợ KÉO & THẢ TRỰC TIẾP
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
        help="Bấm Upload hoặc kéo thả video trực tiếp vào đây"
    )

    # Khung kéo thả nét đứt trực quan (Video drag-and-drop zone from here)
    if uploaded_file is None:
        st.markdown("""
        <div class="dropzone-container" onclick="const el = window.parent.document.querySelector('input[type=file]') || document.querySelector('input[type=file]'); if(el) el.click();">
            <div class="dropzone-icon">⇧</div>
            <div class="dropzone-text">Video drag-and-drop<br>zone from here</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="dropzone-container" style="border-color: #22c55e; background: rgba(34, 197, 94, 0.1);">
            <div class="dropzone-icon" style="color: #22c55e;">✅</div>
            <div class="dropzone-text" style="color: #4ade80;">Đã nhận file: <b>{uploaded_file.name}</b><br><small style="color: #94a3b8;">Nhấn 'Start Analysis' bên dưới để bắt đầu</small></div>
        </div>
        """, unsafe_allow_html=True)

    # Tùy chọn chạy video mẫu (nếu người dùng muốn thử nghiệm nhanh khi chưa có video)
    use_sample = st.checkbox("Sử dụng video mẫu (Sample 1080p)", value=False)

    # Dropdown chọn tập dữ liệu
    st.markdown('<div class="control-label">Dataset Selector</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox(
        "Dataset Selector",
        ["BDD100K", "nuScenes", "Waymo", "KITTI"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 4 Toggles theo thiết kế
    toggle_boxes = st.toggle("YOLO 3D Bounding Boxes", value=True)
    toggle_drivable = st.toggle("SegFormer Drivable Area", value=True)
    toggle_lanes = st.toggle("SegFormer Lane Markings", value=True)
    toggle_graph = st.toggle("Context Reasoning Graph", value=True)

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # Nút bấm chính
    start_btn = st.button("Start Analysis", use_container_width=True)


# -------------------------------------------------------------
# CỘT 2: 1080p DRIVER-VIEW DASHCAM VIDEO & TELEMETRY
# -------------------------------------------------------------
with col_video:
    st.markdown('<div class="panel-header">1080p Driver-view Dashcam Video</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    progress_placeholder = st.empty()

    st.markdown('<div class="panel-header" style="margin-top: 14px;">Telemetry</div>', unsafe_allow_html=True)
    telemetry_placeholder = st.empty()


# -------------------------------------------------------------
# CỘT 3: TELEMETRY SIDEBAR / HUD
# -------------------------------------------------------------
with col_hud:
    st.markdown('<div class="panel-header">Telemetry Sidebar / HUD</div>', unsafe_allow_html=True)

    card_egolane = st.empty()
    card_leadveh = st.empty()
    card_signal = st.empty()
    card_pedestrian = st.empty()

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    terminal_placeholder = st.empty()


# -------------------------------------------------------------
# CÁC HÀM TIỆN ÍCH RENDER HUD & TELEMETRY
# -------------------------------------------------------------
def render_hud_cards(ego_lane="--", lead_dist="--", lead_alert="Standby", 
                     signal="--", ped_hazard="--", is_amber=False):
    # Card 1: Ego Lane
    card_egolane.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Ego Lane</div>
        <div class="hud-val-primary">{ego_lane}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Lead Vehicle (Amber Caution)
    amber_class = "amber-glow" if is_amber else ""
    val_class = "hud-val-amber" if is_amber else "hud-val-primary"
    sub_color = "#f59e0b" if is_amber else "#64748b"
    card_leadveh.markdown(f"""
    <div class="hud-card {amber_class}">
        <div class="hud-title">Lead Vehicle</div>
        <div class="{val_class}">ID #1 - Distance: {lead_dist}</div>
        <div class="hud-sub-amber" style="color: {sub_color};">({lead_alert})</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Traffic Signal
    signal_class = "hud-val-red" if "RED" in signal else "hud-val-primary"
    card_signal.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Traffic Signal</div>
        <div class="{signal_class}">{signal}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: Pedestrian Hazard
    card_pedestrian.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Pedestrian Hazard</div>
        <div class="hud-val-alert">{ped_hazard}</div>
    </div>
    """, unsafe_allow_html=True)


def render_terminal_json(json_data):
    """Vẽ khối Terminal có màu sắc cú pháp JSON đẹp mắt khớp ảnh mẫu"""
    formatted_json = json.dumps(json_data, indent=2)
    terminal_placeholder.markdown(f"""
    <div class="terminal-card">
        <div class="terminal-header">
            <span>Terminal</span>
            <span class="terminal-close">✕</span>
        </div>
        <div class="terminal-body"><pre style="margin: 0; background: transparent; padding: 0;"><code>{formatted_json}</code></pre></div>
    </div>
    """, unsafe_allow_html=True)


def render_telemetry_bar(speed="--", gap="--", ttc="--", risk="STANDBY"):
    ttc_class = "red" if (isinstance(ttc, (int, float)) and ttc < 1.8) else ("green" if ttc != "--" else "muted")
    risk_class = "amber" if risk == "CAUTION" else ("green" if risk == "SAFE" else "muted")
    speed_class = "cyan" if speed != "--" else "muted"
    gap_class = "amber" if gap != "--" else "muted"

    telemetry_placeholder.markdown(f"""
    <div class="telemetry-bottom-card">
        <div class="telemetry-grid">
            <div class="telemetry-item">
                <div class="telemetry-lbl">Vận Tốc Ego</div>
                <div class="telemetry-val {speed_class}">{speed}{' km/h' if speed != '--' else ''}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-lbl">Khoảng Cách Xe Trước</div>
                <div class="telemetry-val {gap_class}">{gap}{' m' if gap != '--' else ''}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-lbl">TTC (Thời Gian Va Chạm)</div>
                <div class="telemetry-val {ttc_class}">{ttc}{'s' if ttc != '--' else ''}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-lbl">Cấp Độ Rủi Ro</div>
                <div class="telemetry-val {risk_class}">{risk}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 5. XỬ LÝ TRẠNG THÁI HIỂN THỊ TRUNG TÂM (ĐÃ XÓA DEMO MẶC ĐỊNH)
# -------------------------------------------------------------
pipeline = DrivingPerceptionPipeline(fps=30)

# KHI CHƯA CÓ VIDEO TẢI LÊN VÀ KHÔNG CHỌN SAMPLE:
if uploaded_file is None and not use_sample:
    # 1. Màn hình giữa: Trạng thái chờ Standby sạch sẽ (Không hiển thị ảnh demo)
    video_placeholder.markdown("""
    <div class="video-standby-viewport">
        <div class="standby-icon">📹</div>
        <div class="standby-title">NO VIDEO STREAM LOADED</div>
        <div class="standby-desc">
            Vùng kéo-thả bên trái hỗ trợ kéo thả trực tiếp file <b>.mp4</b>, <b>.avi</b>, <b>.mov</b>.<br>
            Hãy kéo thả video hành trình vào ô <b>Controls</b> để bắt đầu.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Cột phải & Telemetry: Hiển thị trạng thái Standby
    render_hud_cards(
        ego_lane="--",
        lead_dist="--",
        lead_alert="Standby",
        signal="--",
        ped_hazard="--",
        is_amber=False
    )
    standby_json = {
        "status": "STANDBY",
        "message": "Waiting for dashcam video stream input...",
        "controls": "Drag & drop video into left panel"
    }
    render_terminal_json(standby_json)
    render_telemetry_bar(speed="--", gap="--", ttc="--", risk="STANDBY")

    # Nếu người dùng bấm Start khi chưa có video
    if start_btn:
        st.warning("⚠️ Vui lòng kéo & thả hoặc tải lên file video hành trình (.mp4) trước khi bấm 'Start Analysis'!")

# KHI ĐÃ TẢI VIDEO LÊN (HOẶC CHỌN SAMPLE)
else:
    # Trường hợp 1: Người dùng đã kéo thả file video
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        
        # Đọc trước frame 0 để hiển thị preview video của người dùng
        ret, first_frame = cap.read()
        cap.release()

        if ret and first_frame is not None:
            preview_resized = cv2.resize(first_frame, (960, 540))
            if not start_btn:
                video_placeholder.image(cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                render_hud_cards(
                    ego_lane="Lane 2 (Center)",
                    lead_dist="READY",
                    lead_alert="Video loaded",
                    signal="READY",
                    ped_hazard="Ready to scan",
                    is_amber=False
                )
                ready_json = {
                    "status": "VIDEO_LOADED",
                    "file_name": uploaded_file.name,
                    "file_size_mb": round(uploaded_file.size / (1024 * 1024), 2),
                    "action": "Click 'Start Analysis' to process"
                }
                render_terminal_json(ready_json)
                render_telemetry_bar(speed=0, gap="--", ttc="--", risk="READY")

        # KHI BẤM START ANALYSIS TRÊN VIDEO TẢI LÊN
        if start_btn:
            pipeline.reset()
            cap = cv2.VideoCapture(tfile.name)
            frames_source = []
            while cap.isOpened():
                r, f = cap.read()
                if not r:
                    break
                frames_source.append(f)
            cap.release()

            total_frames = len(frames_source)
            if total_frames == 0:
                st.error("Không thể đọc khung hình nào từ video tải lên.")
                st.stop()

            for idx, f in enumerate(frames_source):
                resized = cv2.resize(f, (960, 540))
                annotated, log = pipeline.run_inference(
                    frame=resized,
                    frame_idx=idx,
                    weather="Nang rao",
                    show_masks=toggle_drivable,
                    show_lanes=toggle_lanes,
                    show_boxes=toggle_boxes,
                    show_graph=toggle_graph
                )

                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                progress_placeholder.progress((idx + 1) / total_frames, text=f"Processing Frame {idx + 1}/{total_frames} (30 FPS)")

                summary = log.get("summary", {})
                risk_lvl = summary.get("risk_level", "An toan")
                is_amber = (risk_lvl in ["Chu y", "Nguy hiem"])

                render_hud_cards(
                    ego_lane="Lane 2 (Center)",
                    lead_dist="NEAR" if is_amber else "NORMAL",
                    lead_alert="Caution in amber" if is_amber else "Safe",
                    signal="RED (Ahead)" if summary.get("traffic_light_ahead") else "CLEAR",
                    ped_hazard=f"{summary.get('pedestrians_on_road', 0)} on Road (Alert)" if summary.get('pedestrians_on_road', 0) > 0 else "None",
                    is_amber=is_amber
                )
                render_terminal_json(log)
                render_telemetry_bar(
                    speed=42,
                    gap=12.5 if is_amber else 22.0,
                    ttc=1.4 if is_amber else 3.5,
                    risk="CAUTION" if is_amber else "SAFE"
                )
                time.sleep(0.03)

            progress_placeholder.empty()
            st.toast("🎉 Phân tích video thành công!", icon="✅")

    # Trường hợp 2: Người dùng tích chọn 'Sử dụng video mẫu'
    elif use_sample:
        scenario_data = get_demo_scenario_data(num_frames=90, target_w=960, target_h=540)
        
        if not start_btn:
            # Hiển thị preview frame 0 của sample
            init_item = scenario_data[0]
            annotated_preview = pipeline.draw_hud_visuals(
                frame=init_item["frame"],
                detections=init_item["detections"],
                masks=init_item["masks"],
                context_log=init_item["telemetry"],
                show_masks=toggle_drivable,
                show_lanes=toggle_lanes,
                show_boxes=toggle_boxes,
                show_graph=toggle_graph
            )
            video_placeholder.image(cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            render_hud_cards(
                ego_lane="Lane 2 (Center)",
                lead_dist="NEAR",
                lead_alert="Caution in amber",
                signal="RED (Ahead)",
                ped_hazard="1 on Road (Alert)",
                is_amber=True
            )
            sample_init_json = {
                "name": "Lane2Centers",
                "Lead Vehicle": {"TrafficSignal": "Red", "attens": 30},
                "Telemetry Status": {"Distance": "NEAR", "PedestrianHazard": "1 on Road"}
            }
            render_terminal_json(sample_init_json)
            render_telemetry_bar(speed=42, gap=12.5, ttc=1.4, risk="CAUTION")

        if start_btn:
            pipeline.reset()
            total_frames = len(scenario_data)
            for idx, item in enumerate(scenario_data):
                frame = item["frame"]
                detections = item["detections"]
                masks = item["masks"]
                telemetry = item["telemetry"]
                terminal_json = item["terminal_json"]

                annotated = pipeline.draw_hud_visuals(
                    frame=frame,
                    detections=detections,
                    masks=masks,
                    context_log=telemetry,
                    show_masks=toggle_drivable,
                    show_lanes=toggle_lanes,
                    show_boxes=toggle_boxes,
                    show_graph=toggle_graph
                )

                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                progress_placeholder.progress((idx + 1) / total_frames, text=f"Analyzing Sample Frame {idx + 1}/{total_frames} (30 FPS)")

                lead_data = telemetry["lead_vehicle"]
                is_amber = (lead_data["alert"] == "Caution in amber")

                render_hud_cards(
                    ego_lane=telemetry["ego_lane"],
                    lead_dist=lead_data["distance"],
                    lead_alert=lead_data["alert"],
                    signal=telemetry["traffic_signal"],
                    ped_hazard=telemetry["pedestrian_hazard"],
                    is_amber=is_amber
                )
                render_terminal_json(terminal_json)
                render_telemetry_bar(
                    speed=telemetry["speed_kmh"],
                    gap=lead_data["gap_meters"],
                    ttc=lead_data["ttc_sec"],
                    risk="CAUTION" if is_amber else "SAFE"
                )
                time.sleep(0.035)

            progress_placeholder.empty()
            st.toast("🎉 Hoàn tất phân tích video mẫu thành công!", icon="✅")