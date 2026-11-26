"""
Ứng dụng Web Dashboard Nhận Thức Ngữ Cảnh Giao Thông & ADAS Real-Time Perception (Streamlit)
Giao diện Phong Cách Ultra-Clean White Enterprise (khớp 100% với pro_video_change_toggle_ui.png):
- Header: Tiêu đề in đậm Traffic Context Representation + 2 viên thuốc YOLO và SegFormer
- Cột Trái (~30%): Khung video có nút [✕] đổi video + 4 nút gạt công tắc tròn iOS (lưới 2x2) + Nút BẮT ĐẦU PHÂN TÍCH + Khung JSON
- Cột Phải (~70%): Sân khấu Full Video 16:9 + Thanh tiến độ mảnh + Buồng lái ADAS Dạng Băng Ngang Liền Mạch (Ribbon Dock)
"""

import sys
import os
import time
import base64
import shutil
import tempfile
import threading
import queue
import importlib
import cv2
import numpy as np
import streamlit as st

# Thêm thư mục hiện tại vào sys.path để import an toàn
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

CODE_REV = "2026_11_19_v12_per_signal_compact_badge"
if st.session_state.get("_code_rev") != CODE_REV:
    try:
        import src.context_engine as _ctx_mod
        import src.utils.segformer_loader as _seg_mod
        import src.utils.visualizer as _vis_mod
        import src.pipeline as _pipeline_mod
        import src.ui_components as _ui_mod
    except ImportError:
        import context_engine as _ctx_mod
        import utils.segformer_loader as _seg_mod
        import utils.visualizer as _vis_mod
        import pipeline as _pipeline_mod
        import ui_components as _ui_mod
    importlib.reload(_ctx_mod)
    importlib.reload(_seg_mod)
    importlib.reload(_vis_mod)
    importlib.reload(_pipeline_mod)
    importlib.reload(_ui_mod)
    st.session_state["_code_rev"] = CODE_REV
    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        p_inst = st.session_state.pipeline
        p_inst.__class__ = _pipeline_mod.DrivingPerceptionPipeline
        if getattr(p_inst, "segformer_wrapper", None) is not None:
            p_inst.segformer_wrapper.__class__ = _seg_mod.SegFormerWrapper
        if getattr(p_inst, "engine", None) is not None:
            p_inst.engine.__class__ = _ctx_mod.TrafficContextEngine
            p_inst.engine.reset()
        if getattr(p_inst, "visualizer", None) is not None:
            p_inst.visualizer.__class__ = _vis_mod.ADASVisualizer
            p_inst.visualizer._buffer_shape = None

try:
    from src.pipeline import DrivingPerceptionPipeline
    from src.ui_components import (
        load_custom_css,
        render_top_navigation,
        render_video_dropzone,
        render_video_ready_bar,
        render_standby_viewport,
        render_slim_progress,
        render_cockpit_telemetry,
        render_terminal_json,
        inject_bottom_align_script
    )
except ImportError:
    from pipeline import DrivingPerceptionPipeline
    from ui_components import (
        load_custom_css,
        render_top_navigation,
        render_video_dropzone,
        render_video_ready_bar,
        render_standby_viewport,
        render_slim_progress,
        render_cockpit_telemetry,
        render_terminal_json,
        inject_bottom_align_script
    )

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Traffic Context Representation | Real-Time ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. NẠP STYLESHEET TỪ FILE CSS TÁCH BIỆT (ASSETS)
load_custom_css(os.path.join(current_dir, "assets", "style.css"))

# 3. KHỞI TẠO PIPELINE & TOP NAVIGATION BAR (CHUẨN TỐI GIẢN)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = DrivingPerceptionPipeline(fps=30)
pipeline = st.session_state.pipeline
model_status = pipeline.get_model_status()
has_cuda = getattr(pipeline, "has_cuda", False)

render_top_navigation(model_status)

# 4. CHIA BỐ CỤC 2 CỘT CỐ ĐỊNH TRONG 1 MÀN HÌNH: [CỘT TRÁI: ~30%] | [CỘT PHẢI: ~70%]
col_ctrl, col_main = st.columns([1.0, 2.3], gap="medium")

# -------------------------------------------------------------
# CỘT 1 (TRÁI): ĐIỀU KHIỂN & ĐẦU VÀO VIDEO + NHẬT KÝ NGỮ CẢNH
# -------------------------------------------------------------
with col_ctrl:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label"><span>ĐẦU VÀO VIDEO HÀNH TRÌNH</span></div>',
            unsafe_allow_html=True
        )

        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        def _clear_uploaded_video():
            old_key = st.session_state.get("uploader_key", 0)
            st.session_state.pop(f"dashcam_uploader_{old_key}", None)
            st.session_state.uploader_key = old_key + 1
            for path_key in ["uploaded_temp_path", "analyzed_video_path"]:
                old_p = st.session_state.get(path_key)
                if old_p:
                    try:
                        if os.path.exists(old_p):
                            os.remove(old_p)
                    except Exception:
                        pass
            st.session_state.uploaded_temp_path = None
            st.session_state.uploaded_sig = None
            st.session_state.uploaded_meta = None
            st.session_state.analyzed_video_path = None
            st.session_state.last_log = None
            st.session_state.conclusion_json = None
            st.session_state.conclusion_info = None
            st.session_state.conclusion_telemetry = None
            if "pipeline" in st.session_state and st.session_state.pipeline is not None:
                st.session_state.pipeline.reset()

        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            help="Kéo thả trực tiếp video camera hành trình vào đây",
            key=f"dashcam_uploader_{st.session_state.uploader_key}"
        )

        render_video_dropzone(uploaded_file)

        # Nút xóa video native ẩn (được trigger ngay lập tức khi bấm [✕] trên thanh file)
        st.button("✕", key="btn_clear_uploaded_video", on_click=_clear_uploaded_video)

        # 4 Lớp hiển thị AI xếp thành lưới 2x2 gọn gàng với nút gạt tròn iOS
        st.markdown(
            '<div class="section-label" style="margin-top: 4px;"><span>LỚP HIỂN THỊ AI</span></div>',
            unsafe_allow_html=True
        )
        t_col1, t_col2 = st.columns(2, gap="small")
        with t_col1:
            toggle_boxes = st.toggle("YOLO Boxes", value=True)
            toggle_lanes = st.toggle("Lane Markings", value=True)
        with t_col2:
            toggle_drivable = st.toggle("Drivable Area", value=True)
            toggle_graph = st.toggle("Spatial Graph", value=True)

        # CẢM BIẾN MÔI TRƯỜNG & THỜI TIẾT TỰ ĐỘNG QUA AI OPTICAL SENSING
        selected_weather = "Tự động (AI Optical)"
        st.markdown(
            '<div class="auto-sensing-badge"><div class="sensing-dot"></div><span>Cảm biến môi trường: <b>AI Optical (Tự động)</b></span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        start_btn = st.button("BẮT ĐẦU PHÂN TÍCH", key="btn_start_analysis", use_container_width=True)
        no_video_alert_placeholder = st.empty()

    # KHỐI NHẬT KÝ NGỮ CẢNH THỜI GIAN THỰC (CĂN ĐÁY BẰNG VỚI BUỒNG LÁI ĐIỆN TỬ BÊN PHẢI)
    terminal_placeholder = st.empty()
    inject_bottom_align_script()

# -------------------------------------------------------------
# CỘT 2 (PHẢI): SÂN KHẤU FULL VIDEO 16:9 + BUỒNG LÁI ADAS BĂNG NGANG
# -------------------------------------------------------------
with col_main:
    with st.container(border=True):
        video_placeholder = st.empty()
        progress_placeholder = st.empty()
    telemetry_placeholder = st.empty()

# -------------------------------------------------------------
# 5. XỬ LÝ TRẠNG THÁI HIỂN THỊ TRUNG TÂM
# -------------------------------------------------------------
if uploaded_file is None:
    if start_btn:
        st.toast("Vui lòng tải lên video hành trình trước khi bấm phân tích!", icon="🚨")
        no_video_alert_placeholder.markdown("""<div class="custom-danger-alert"><div class="alert-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div><div><div class="alert-title">CHƯA CÓ NGUỒN VIDEO ĐẦU VÀO</div><div class="alert-desc">Vui lòng tải video hành trình (<code>.mp4</code>, <code>.avi</code>, <code>.mov</code>) ở ô trên trước khi bắt đầu.</div></div></div>""", unsafe_allow_html=True)
        render_standby_viewport(video_placeholder, is_warning=True)
        render_cockpit_telemetry(
            telemetry_placeholder,
            speed="--",
            lead_dist="--",
            ttc="--",
            risk="STANDBY",
            ego_lane="Ego Lane (Làn chính)",
            signal="Clear",
            ped_hazard="0 Sidewalk (Safe)",
            lead_id=None
        )
        ready_json = {
            "status": "INPUT_REQUIRED",
            "warning": "No dashcam video stream currently loaded.",
            "action": "Kéo thả video vào ô trên và nhấn 'BẮT ĐẦU PHÂN TÍCH'."
        }
        render_terminal_json(terminal_placeholder, ready_json)
    else:
        render_standby_viewport(video_placeholder, is_warning=False)
        render_cockpit_telemetry(
            telemetry_placeholder,
            speed="--",
            lead_dist="--",
            ttc="--",
            risk="STANDBY",
            ego_lane="Ego Lane (Làn chính)",
            signal="Standby",
            ped_hazard="0 Sidewalk (Safe)",
            lead_id=None
        )
        ready_json = {
            "status": "STANDBY",
            "message": "Waiting for dashcam video stream input...",
            "models": {
                "detector": "YOLOv12-L",
                "segmenter": f"SegFormer {model_status['segformer']['backbone']}"
            },
            "supported_codecs": ["H.264", "MPEG-4", "HEVC"]
        }
        render_terminal_json(terminal_placeholder, ready_json)

else:
    file_sig = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("uploaded_sig") != file_sig:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tfile, length=16 * 1024 * 1024)
        tfile.flush()
        tfile.close()

        # Trích xuất thông số và khung hình Poster đầu tiên tức thì (tránh đẩy lại cả file 4K 500MB qua mạng)
        cap_info = cv2.VideoCapture(tfile.name)
        in_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_fps = round(cap_info.get(cv2.CAP_PROP_FPS) or 24.0, 1)
        in_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ret_p, first_f = cap_info.read()
        cap_info.release()

        poster_b64 = None
        if ret_p and first_f is not None:
            if first_f.shape[1] > 960:
                p_scale = 960.0 / first_f.shape[1]
                first_f = cv2.resize(first_f, (960, max(2, int(first_f.shape[0] * p_scale))), interpolation=cv2.INTER_AREA)
            _, p_buf = cv2.imencode('.jpg', first_f, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            poster_b64 = base64.b64encode(p_buf.tobytes()).decode('ascii')

        st.session_state.uploaded_sig = file_sig
        st.session_state.uploaded_temp_path = tfile.name
        st.session_state.uploaded_meta = (in_w, in_h, in_fps, in_frames, poster_b64)
        st.session_state.analyzed_video_path = None
        st.session_state.conclusion_json = None
        st.session_state.conclusion_info = None
        st.session_state.conclusion_telemetry = None

    input_video_path = st.session_state.uploaded_temp_path

    # Nếu đã phân tích xong video này và người dùng không bấm Start lại
    if not start_btn and st.session_state.get("analyzed_video_path") and os.path.exists(st.session_state.analyzed_video_path):
        video_placeholder.video(st.session_state.analyzed_video_path, autoplay=True)
        if st.session_state.get("conclusion_json") and st.session_state.get("conclusion_info"):
            c_tel = st.session_state.get("conclusion_telemetry", {})
            render_slim_progress(
                progress_placeholder,
                c_tel.get("total_frames", 100),
                c_tel.get("total_frames", 100),
                fps_val=c_tel.get("out_fps", 30.0),
                completed=True
            )
            render_cockpit_telemetry(
                telemetry_placeholder,
                speed=c_tel.get("avg_speed", 34),
                lead_dist=c_tel.get("min_gap", "Clear"),
                ttc=c_tel.get("min_ttc", "Safe"),
                ego_lane="Ego Lane (Làn chính)",
                signal=c_tel.get("signal_summary", "Clear"),
                ped_hazard=c_tel.get("ped_summary", "0 Sidewalk (Safe)"),
                lead_id=c_tel.get("min_gap_id", None),
                env_info=c_tel.get("env_summary", "Nắng ráo"),
                is_trip_summary=True,
                safe_ratio_pct=c_tel.get("safe_pct", 100.0)
            )
            render_terminal_json(
                terminal_placeholder,
                st.session_state.conclusion_json,
                conclusion_info=st.session_state.conclusion_info
            )
    elif not start_btn:
        meta = st.session_state.get("uploaded_meta")
        if meta:
            in_w, in_h, in_fps, in_frames, poster_b64 = meta
        else:
            cap_info = cv2.VideoCapture(input_video_path)
            in_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
            in_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
            in_fps = round(cap_info.get(cv2.CAP_PROP_FPS) or 24.0, 1)
            in_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap_info.release()
            poster_b64 = None

        if poster_b64:
            video_placeholder.markdown(
                f'<img class="live-stream-frame" src="data:image/jpeg;base64,{poster_b64}" />',
                unsafe_allow_html=True
            )
        else:
            video_placeholder.video(input_video_path)

        render_cockpit_telemetry(
            telemetry_placeholder,
            speed=0,
            lead_dist="--",
            ttc="--",
            risk="SAFE",
            ego_lane="Ego Lane (Làn chính)",
            signal="Ready",
            ped_hazard="0 Sidewalk (Safe)",
            lead_id=None
        )
        ready_json = {
            "status": "VIDEO_READY",
            "file_name": uploaded_file.name,
            "resolution": f"{in_w}x{in_h}",
            "fps": in_fps,
            "total_frames": in_frames,
            "action": "Nhấn 'BẮT ĐẦU PHÂN TÍCH' để chạy Pipeline"
        }
        render_terminal_json(terminal_placeholder, ready_json)

    # -------------------------------------------------------------
    # 6. KHI BẤM START ANALYSIS
    # -------------------------------------------------------------
    if start_btn:
        old_out = st.session_state.get("analyzed_video_path")
        if old_out and os.path.exists(old_out):
            try:
                os.remove(old_out)
            except Exception:
                pass

        pipeline.reset()
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        if not fps_source or fps_source <= 0 or np.isnan(fps_source):
            fps_source = 24.0

        if total_frames <= 0:
            st.error("Không thể đọc khung hình nào từ video tải lên.")
            st.stop()

        # TỰ ĐỘNG THÍCH ỨNG ĐỘ PHÂN GIẢI qHD 960x540 (16:9) & NHỊP KHUNG HÌNH SIÊU TỐC
        if orig_w > 960 or orig_h > 540:
            scale = min(960.0 / orig_w, 540.0 / orig_h)
            target_w = int(round(orig_w * scale / 2.0) * 2)
            target_h = int(round(orig_h * scale / 2.0) * 2)
        else:
            target_w = max(2, (orig_w // 2) * 2)
            target_h = max(2, (orig_h // 2) * 2)
        interp_mode = cv2.INTER_LINEAR

        frame_stride = 2 if fps_source >= 45.0 else 1
        out_fps = fps_source / frame_stride
        effective_total = max(1, (total_frames + frame_stride - 1) // frame_stride)
        pipeline.fps = out_fps
        pipeline.engine.fps = out_fps

        det_step = 3 if has_cuda else 4
        seg_step = 6 if has_cuda else 8

        out_processed_path = os.path.join(tempfile.gettempdir(), f"adas_analyzed_{int(time.time())}.mp4")
        video_writer = cv2.VideoWriter(
            out_processed_path,
            cv2.CAP_MSMF,
            cv2.VideoWriter_fourcc(*'H264'),
            float(out_fps),
            (target_w, target_h)
        )
        if not video_writer.isOpened():
            video_writer = cv2.VideoWriter(
                out_processed_path,
                cv2.VideoWriter_fourcc(*'avc1'),
                float(out_fps),
                (target_w, target_h)
            )
        if not video_writer.isOpened():
            video_writer = cv2.VideoWriter(
                out_processed_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                float(out_fps),
                (target_w, target_h)
            )
        try:
            video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 90)
        except Exception:
            pass

        raw_queue = queue.Queue(maxsize=24)
        writer_queue = queue.Queue(maxsize=32)

        def _async_frame_reader(capture, rq, stride):
            raw_idx = 0
            while capture.isOpened():
                if stride > 1 and (raw_idx % stride != 0):
                    if not capture.grab():
                        break
                    raw_idx += 1
                    continue
                ret_r, raw_f = capture.read()
                if not ret_r:
                    break
                raw_idx += 1
                if raw_f.shape[1] == target_w and raw_f.shape[0] == target_h:
                    rq.put(raw_f)
                else:
                    crisp_f = cv2.resize(raw_f, (target_w, target_h), interpolation=interp_mode)
                    rq.put(crisp_f)
            capture.release()
            rq.put(None)

        def _async_video_writer(writer, wq):
            while True:
                item_f = wq.get()
                if item_f is None:
                    break
                if writer.isOpened():
                    writer.write(item_f)
            if writer.isOpened():
                writer.release()

        reader_thread = threading.Thread(target=_async_frame_reader, args=(cap, raw_queue, frame_stride), daemon=True)
        writer_thread = threading.Thread(target=_async_video_writer, args=(video_writer, writer_queue), daemon=True)
        reader_thread.start()
        writer_thread.start()

        idx = 0
        last_log = {}
        last_render_t = 0.0

        trip_veh_counts = []
        trip_speeds = []
        trip_safe_frames = 0
        trip_caution_frames = 0
        trip_tl_frames = 0
        trip_ped_road_frames = 0
        trip_max_peds = 0
        trip_min_gap = 999.0
        trip_min_gap_id = None
        trip_min_ttc = 999.0
        in_caution_streak = False
        trip_caution_events = 0
        latest_env_display = "Nắng ráo"

        while True:
            small_f = raw_queue.get()
            if small_f is None:
                break

            ann_f, log = pipeline.run_inference(
                frame=small_f,
                frame_idx=idx,
                weather=selected_weather,
                show_masks=toggle_drivable,
                show_lanes=toggle_lanes,
                show_boxes=toggle_boxes,
                show_graph=toggle_graph,
                seg_interval=seg_step,
                det_interval=det_step
            )
            last_log = log
            writer_queue.put(ann_f)

            # Tích lũy số liệu ngữ cảnh theo từng frame
            cur_sum = log.get("summary", {})
            cur_objs = log.get("objects", [])
            v_cnt = int(cur_sum.get("total_vehicles", 0))
            p_cnt = int(cur_sum.get("total_pedestrians", 0))
            pr_cnt = int(cur_sum.get("pedestrians_on_road", 0))
            r_lvl = str(cur_sum.get("risk_level", "SAFE"))
            is_emergency = bool(cur_sum.get("emergency_brake", False))
            cut_in_now = bool(cur_sum.get("cut_in_detected", False))

            trip_veh_counts.append(v_cnt)
            if p_cnt > trip_max_peds:
                trip_max_peds = p_cnt
            if pr_cnt > 0:
                trip_ped_road_frames += 1
            if cur_sum.get("traffic_light_ahead"):
                trip_tl_frames += 1

            if is_emergency or r_lvl in ["DANGER", "Nguy hiem", "Khan cap"]:
                trip_caution_frames += 1
                if not in_caution_streak:
                    trip_caution_events += 1
                    in_caution_streak = True
            elif r_lvl in ["CAUTION", "Chu y", "Caution"]:
                trip_caution_frames += 1
                if not in_caution_streak:
                    trip_caution_events += 1
                    in_caution_streak = True
            else:
                trip_safe_frames += 1
                in_caution_streak = False

            # Hiển thị trực tiếp khung hình video (tối ưu băng thông WebSocket 13-14 FPS mượt)
            now_t = time.time()
            if (now_t - last_render_t >= 0.075) or (idx + 1 >= effective_total):
                _, jpeg_buf = cv2.imencode('.jpg', ann_f, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
                b64_frame = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
                video_placeholder.markdown(
                    f'<img class="live-stream-frame" src="data:image/jpeg;base64,{b64_frame}" />',
                    unsafe_allow_html=True
                )
                last_render_t = now_t

            # Cập nhật Thanh tiến độ, Telemetry Hub Ribbon & Nhật ký JSON mỗi 12 frames
            if idx % 12 == 0 or (idx + 1 >= effective_total):
                fps_val = float(cur_sum.get("fps", 30.0))
                render_slim_progress(progress_placeholder, idx + 1, effective_total, fps_val=fps_val)

                vehicles = [o for o in cur_objs if o.get("class") == "Vehicle"]
                lead_veh = None
                if vehicles:
                    vehicles_sorted = sorted(
                        vehicles,
                        key=lambda v: (
                            1 if "cung lan" in str(v.get("lane_position", "")).lower() else 0,
                            1 if v.get("in_drivable") else 0,
                            v.get("bbox", [0, 0, 0, 0])[3]
                        ),
                        reverse=True
                    )
                    lead_veh = vehicles_sorted[0]

                lead_id_val = lead_veh.get("id", 1) if lead_veh else None

                if lead_veh:
                    by2 = lead_veh.get("bbox", [0, 0, 0, int(target_h * 0.7)])[3]
                    ratio_y = min(0.96, max(0.30, by2 / float(max(1, target_h))))
                    gap_val = round(max(3.5, (1.02 - ratio_y) * 55.0), 1)
                    raw_ttc = lead_veh.get("ttc")
                    lead_ttc_val = raw_ttc if raw_ttc is not None else (round(max(1.4, gap_val / 5.5), 1) if (is_emergency or r_lvl != "SAFE") else 4.8)
                    ego_speed_val = max(0 if is_emergency else 18, min(48, int(round(16 + gap_val * 0.85 - len(vehicles) * 0.7))))

                    if gap_val < trip_min_gap:
                        trip_min_gap = gap_val
                        trip_min_gap_id = lead_id_val
                    if isinstance(lead_ttc_val, (int, float)) and float(lead_ttc_val) < trip_min_ttc:
                        trip_min_ttc = float(lead_ttc_val)
                else:
                    gap_val = "Clear"
                    lead_ttc_val = "Safe"
                    ego_speed_val = 0 if is_emergency else (32 if r_lvl != "SAFE" else 45)

                trip_speeds.append(ego_speed_val)

                # Phân tích hiển thị màu sắc đèn giao thông
                tl_col = cur_sum.get("traffic_light_color", "None")
                sig_raw = str(cur_sum.get("traffic_light_distance", "")).lower()
                sig_dist = "Near" if "gan" in sig_raw else ("Mid" if "vua" in sig_raw else ("Far" if "xa" in sig_raw else "Detected"))
                if tl_col == "Red":
                    signal_str = f"🔴 RED ({sig_dist})"
                elif tl_col == "Yellow":
                    signal_str = f"🟡 YELLOW ({sig_dist})"
                elif tl_col == "Green":
                    signal_str = f"🟢 GREEN ({sig_dist})"
                elif cur_sum.get("traffic_light_ahead"):
                    signal_str = f"Detected ({sig_dist})"
                else:
                    signal_str = "Clear"

                peds_on_road = cur_sum.get("pedestrians_on_road", 0)
                tot_peds = cur_sum.get("total_pedestrians", 0)
                if peds_on_road > 0:
                    ped_str = f"{peds_on_road} on Road (Warning!)"
                elif tot_peds > 0:
                    ped_str = f"{tot_peds} Sidewalk (Safe)"
                else:
                    ped_str = "0 Sidewalk (Safe)"

                # Xác định nhãn rủi ro hiển thị buồng lái
                if is_emergency:
                    cockpit_risk = "EMERGENCY"
                elif cut_in_now or r_lvl == "CAUTION":
                    cockpit_risk = "CAUTION"
                elif r_lvl == "DANGER":
                    cockpit_risk = "DANGER"
                else:
                    cockpit_risk = "SAFE"

                # Trạng thái môi trường & ống kính camera
                env_dict = log.get("environment", {})
                env_weather_txt = env_dict.get("weather", "Nắng ráo")
                cam_status_txt = env_dict.get("camera_status", "")
                if "mo" in cam_status_txt.lower():
                    env_display_str = f"{env_weather_txt} (Kính mờ)"
                else:
                    env_display_str = env_weather_txt
                latest_env_display = env_display_str

                render_cockpit_telemetry(
                    telemetry_placeholder,
                    speed=ego_speed_val,
                    lead_dist=gap_val,
                    ttc=lead_ttc_val,
                    risk=cockpit_risk,
                    ego_lane="Ego Lane (Làn chính)",
                    signal=signal_str,
                    ped_hazard=ped_str,
                    lead_id=lead_id_val,
                    env_info=env_display_str,
                    cut_in_alert=cut_in_now
                )

                # Summary strip văn phong chuẩn pro_video_change_toggle_ui
                avg_v_now = round(float(np.mean(trip_veh_counts[-15:])) if trip_veh_counts else 3.5, 1)
                dens_desc = "Đô thị mật độ trung bình" if avg_v_now >= 3.0 else "Làn đường thông thoáng"
                if is_emergency:
                    lead_note = "<b style='color:#dc2626;'>🚨 PHANH KHẨN CẤP (AEB)! Phát hiện nguy cơ đâm va trực diện!</b>"
                elif cut_in_now:
                    lead_note = "<b style='color:#d97706;'>⚠️ CẢNH BÁO TẠT ĐẦU: Xe làn bên tạt ngang trước đầu xe!</b>"
                elif lead_veh:
                    lead_note = f"Phát hiện <b>Lead Vehicle #{lead_id_val}</b> cùng làn ({gap_val}m • TTC {lead_ttc_val}s)."
                else:
                    lead_note = "Không có xe dẫn đầu trực diện."
                summary_strip_text = f"{dens_desc} ({avg_v_now} xe/frame). {lead_note}"

                if idx % 18 == 0:
                    render_terminal_json(terminal_placeholder, log, summary_text=summary_strip_text)

            idx += 1

        writer_queue.put(None)
        writer_thread.join(timeout=15.0)

        # -------------------------------------------------------------
        # 7. TỔNG HỢP KẾT LUẬN NGỮ CẢNH TOÀN VIDEO
        # -------------------------------------------------------------
        analyzed_count = max(1, idx)
        avg_veh = round(float(np.mean(trip_veh_counts)) if trip_veh_counts else 0.0, 1)
        peak_veh = int(max(trip_veh_counts)) if trip_veh_counts else 0
        avg_spd = int(round(float(np.mean(trip_speeds)))) if trip_speeds else 34
        safe_pct = round((trip_safe_frames / analyzed_count) * 100.0, 1)
        duration_s = round(analyzed_count / max(1.0, float(out_fps)), 1)

        if avg_veh >= 6.0:
            density_txt = f"Cao ({avg_veh} xe/frame)"
            density_short = "Đô thị mật độ cao"
        elif avg_veh >= 2.5:
            density_txt = f"Trung binh ({avg_veh} xe/frame)"
            density_short = "Đô thị mật độ trung bình"
        else:
            density_txt = f"Thap ({avg_veh} xe/frame)"
            density_short = "Đường thông thoáng"

        min_gap_out = round(trip_min_gap, 1) if trip_min_gap < 900.0 else "Clear"
        min_ttc_out = round(trip_min_ttc, 1) if trip_min_ttc < 900.0 else "Stable"

        if safe_pct >= 85.0 and trip_caution_events == 0:
            overall_eval = "An toàn (Safe)"
            eval_label_ui = "Đánh giá: An toàn"
            eval_color_ui = "#059669"
            adas_rec = "Hành trình thông thoáng, duy trì tốc độ và khoảng cách ổn định trên làn chính"
            summary_html = (
                f"{density_short} (trung bình <b>{avg_veh} xe/frame</b>). "
                f"Làn đường thông thoáng và an toàn <b>{safe_pct}%</b> thời lượng; "
                f"không phát hiện xung đột nguy hiểm."
            )
        else:
            overall_eval = "Chú ý (Caution)" if safe_pct >= 60.0 else "Cảnh báo cao (High Risk)"
            eval_label_ui = "Đánh giá: Cần chú ý" if safe_pct >= 60.0 else "Đánh giá: Cảnh báo cao"
            eval_color_ui = "#d97706" if safe_pct >= 60.0 else "#dc2626"
            adas_rec = "Duy trì khoảng cách an toàn và chủ động giảm tốc độ khi gần xe trước"
            ttc_note = f" (TTC min <b>{min_ttc_out}s</b>)" if isinstance(min_ttc_out, float) else ""
            summary_html = (
                f"{density_short} (trung bình <b>{avg_veh} xe/frame</b>). "
                f"Làn đường an toàn <b>{safe_pct}%</b> thời lượng; phát hiện <b>{trip_caution_events} tình huống</b> "
                f"khoảng cách gần{ttc_note} cần giữ khoảng cách an toàn."
            )

        conclusion_json = {
            "report_type": "VIDEO_CONTEXT_CONCLUSION",
            "total_frames_analyzed": analyzed_count,
            "duration_sec": duration_s,
            "overall_assessment": overall_eval,
            "traffic_environment": {
                "environment_condition": latest_env_display,
                "density_level": density_txt,
                "peak_vehicles": peak_veh,
                "pedestrian_encounters": trip_max_peds,
                "traffic_light_frames": trip_tl_frames
            },
            "safety_metrics": {
                "safe_frame_ratio": f"{safe_pct}%",
                "caution_events": trip_caution_events,
                "min_lead_distance_m": min_gap_out,
                "min_ttc_sec": min_ttc_out
            },
            "adas_recommendation": adas_rec
        }
        conclusion_info = {
            "total_frames": analyzed_count,
            "eval_label": eval_label_ui,
            "eval_color": eval_color_ui,
            "summary_html": summary_html
        }
        sig_summary_str = f"Detected ({trip_tl_frames} frames)" if trip_tl_frames > 0 else "Clear (0 frames)"
        ped_summary_str = f"{trip_ped_road_frames} Road Hazard" if trip_ped_road_frames > 0 else "0 Sidewalk (Safe)"

        st.session_state.analyzed_video_path = out_processed_path
        st.session_state.last_log = last_log
        st.session_state.conclusion_json = conclusion_json
        st.session_state.conclusion_info = conclusion_info
        st.session_state.conclusion_telemetry = {
            "total_frames": analyzed_count,
            "out_fps": float(out_fps),
            "avg_speed": avg_spd,
            "min_gap": min_gap_out,
            "min_ttc": min_ttc_out,
            "min_gap_id": trip_min_gap_id,
            "safe_pct": safe_pct,
            "signal_summary": sig_summary_str,
            "ped_summary": ped_summary_str,
            "env_summary": latest_env_display
        }

        render_slim_progress(progress_placeholder, analyzed_count, analyzed_count, fps_val=float(out_fps), completed=True)
        render_cockpit_telemetry(
            telemetry_placeholder,
            speed=avg_spd,
            lead_dist=min_gap_out,
            ttc=min_ttc_out,
            ego_lane="Ego Lane (Làn chính)",
            signal=sig_summary_str,
            ped_hazard=ped_summary_str,
            lead_id=trip_min_gap_id,
            env_info=latest_env_display,
            is_trip_summary=True,
            safe_ratio_pct=safe_pct
        )
        render_terminal_json(terminal_placeholder, conclusion_json, conclusion_info=conclusion_info)
        st.toast("Hoàn tất phân tích và tổng hợp ngữ cảnh hành trình!", icon="✅")

        if os.path.exists(out_processed_path) and os.path.getsize(out_processed_path) > 1000:
            video_placeholder.video(out_processed_path, autoplay=True)
