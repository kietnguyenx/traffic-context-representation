"""
Module Thành Phần Giao Diện (UI Components)
Traffic Context Representation & ADAS Real-Time Perception Web Dashboard
Khớp 100% với bản thiết kế pro_video_change_toggle_ui.png:
- Header tối giản: Tiêu đề in đậm Traffic Context Representation, 2 viên thuốc trạng thái YOLO và SegFormer
- Cột trái: Đầu vào video có nút [✕] đổi video, 4 nút gạt công tắc tròn iOS (lưới 2x2), Nhật ký JSON nền sáng
- Cột phải: Video 16:9 sắc nét, thanh tiến độ mảnh, Cụm đồng hồ ADAS dạng Băng Ngang Liền Mạch (Ribbon Dock)
"""

import os
import re
import html
import json
import streamlit as st
import streamlit.components.v1 as components


def _c(html_str: str) -> str:
    """Loại bỏ hoàn toàn các khoảng trắng đầu dòng và dòng trống để tránh lỗi CommonMark của Streamlit biến HTML thành code block"""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "".join(lines)


def load_custom_css(css_path: str):
    """Đọc file CSS bên ngoài và nạp an toàn vào giao diện Streamlit"""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_top_navigation(model_status: dict):
    """
    Render thanh Top Navigation Bar chuẩn tối giản:
    - Bỏ hoàn toàn logo [AD] màu xanh chữ trắng
    - Bỏ phụ đề / Real-Time ADAS Perception
    - Bỏ viên thuốc FPS cố định (FPS thật hiển thị động ở thanh tiến độ video)
    """
    yolo_loaded = model_status.get('yolo', {}).get('loaded', False)
    yolo_dot = '<span class="dot-green"></span>' if yolo_loaded else '<span class="dot-amber"></span>'
    yolo_label = "YOLOv12-L" if yolo_loaded else "YOLO Standby"

    seg_loaded = model_status.get('segformer', {}).get('loaded', False)
    seg_dot = '<span class="dot-green"></span>' if seg_loaded else '<span class="dot-amber"></span>'
    seg_label = "SegFormer MiT-B2" if seg_loaded else "SegFormer Standby"

    h = f"""
    <div class="top-header">
        <div class="header-title">Traffic Context Representation</div>
        <div class="header-status">
            <div class="status-dot-pill">{yolo_dot}{yolo_label}</div>
            <div class="status-dot-pill">{seg_dot}{seg_label}</div>
        </div>
    </div>
    """
    st.markdown(_c(h), unsafe_allow_html=True)


def render_video_ready_bar(uploaded_file):
    """
    Render thanh trạng thái file video đã sẵn sàng chuẩn Ultra-Clean
    Nút xóa [✕] nằm trọn vẹn bên trong khung viền (flexbox căn giữa hoàn hảo).
    """
    safe_name = html.escape(uploaded_file.name)
    h = f"""
    <div class="file-ready-bar">
        <div class="file-info-group">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            <span class="file-name" title="{safe_name}">{safe_name}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="file-badge">Đã sẵn sàng</span>
            <button type="button" id="btn-clear-uploaded-video" class="btn-remove-video" title="Xóa video đã nạp">✕</button>
        </div>
    </div>
    """
    st.markdown(_c(h), unsafe_allow_html=True)


def render_video_dropzone(uploaded_file):
    """
    Render khung Drag-and-Drop tải video hoặc thanh trạng thái file
    """
    if uploaded_file is None:
        h = """
        <div id="custom-video-dropzone" class="dropzone-container">
            <div class="dropzone-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
            </div>
            <div class="dropzone-text">Kéo thả hoặc nhấn để chọn file video</div>
            <div class="dropzone-sub">Hỗ trợ MP4, AVI, MOV (Full HD &amp; 4K)</div>
        </div>
        """
        st.markdown(_c(h), unsafe_allow_html=True)
    else:
        render_video_ready_bar(uploaded_file)

    parent_bridge_js = """
(function() {
    var doc = document;
    if (window.__adasUploaderHandlers) {
        var prev = window.__adasUploaderHandlers;
        doc.removeEventListener('click', prev.onClick, true);
        doc.removeEventListener('dragenter', prev.onDragOver, true);
        doc.removeEventListener('dragover', prev.onDragOver, true);
        doc.removeEventListener('dragleave', prev.onDragLeave, true);
        doc.removeEventListener('dragend', prev.onDragLeave, true);
        doc.removeEventListener('drop', prev.onDrop, true);
    }

    function findNativeClearBtn() {
        var btn = doc.querySelector('.st-key-btn_clear_uploaded_video button');
        if (btn) return btn;
        var buttons = doc.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            var txt = (buttons[i].innerText || buttons[i].textContent || '').trim();
            if (txt === '✕') return buttons[i];
        }
        return null;
    }

    function onClick(e) {
        var clearBtn = e.target && e.target.closest ? e.target.closest('#btn-clear-uploaded-video') : null;
        if (clearBtn) {
            e.preventDefault();
            e.stopPropagation();
            var nativeBtn = findNativeClearBtn();
            if (nativeBtn) nativeBtn.click();
            return;
        }
        var customDz = e.target && e.target.closest ? e.target.closest('#custom-video-dropzone') : null;
        if (customDz) {
            e.preventDefault();
            var stInput = doc.querySelector('[data-testid="stFileUploaderDropzoneInput"]') || doc.querySelector('input[type="file"]');
            if (stInput) {
                stInput.value = '';
                stInput.click();
            }
        }
    }

    function onDragOver(e) {
        var customDz = e.target && e.target.closest ? e.target.closest('#custom-video-dropzone') : null;
        if (!customDz) return;
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
        customDz.classList.add('drag-over');
    }

    function onDragLeave(e) {
        var customDz = e.target && e.target.closest ? e.target.closest('#custom-video-dropzone') : null;
        if (!customDz) return;
        e.preventDefault();
        e.stopPropagation();
        customDz.classList.remove('drag-over');
    }

    function onDrop(e) {
        var customDz = e.target && e.target.closest ? e.target.closest('#custom-video-dropzone') : null;
        if (!customDz) return;
        e.preventDefault();
        e.stopPropagation();
        customDz.classList.remove('drag-over');
        var files = e.dataTransfer ? e.dataTransfer.files : null;
        if (!files || files.length === 0) return;

        var stDz = doc.querySelector('[data-testid="stFileUploaderDropzone"]');
        if (!stDz) return;
        var dt = new DataTransfer();
        for (var i = 0; i < files.length; i++) {
            dt.items.add(files[i]);
        }
        var dropEvent = new DragEvent('drop', {
            bubbles: true,
            cancelable: true,
            dataTransfer: dt
        });
        stDz.dispatchEvent(dropEvent);
    }

    window.__adasUploaderBridgeRev = 'v8';
    window.__adasUploaderHandlers = {
        onClick: onClick,
        onDragOver: onDragOver,
        onDragLeave: onDragLeave,
        onDrop: onDrop
    };

    doc.addEventListener('click', onClick, true);
    doc.addEventListener('dragenter', onDragOver, true);
    doc.addEventListener('dragover', onDragOver, true);
    doc.addEventListener('dragleave', onDragLeave, true);
    doc.addEventListener('dragend', onDragLeave, true);
    doc.addEventListener('drop', onDrop, true);
})();
"""
    state_tag = "ready" if uploaded_file is not None else "empty"
    components.html(f"""
    <!-- uploader_state: {state_tag} -->
    <script>
    (function() {{
        var pWin = window.parent;
        var pDoc = pWin.document;
        var existing = pDoc.getElementById('adas-parent-uploader-bridge');
        if (!existing || pWin.__adasUploaderBridgeRev !== 'v8') {{
            if (existing) existing.remove();
            var s = pDoc.createElement('script');
            s.id = 'adas-parent-uploader-bridge';
            s.textContent = {json.dumps(parent_bridge_js)};
            pDoc.head.appendChild(s);
        }}
    }})();
    </script>
    """, height=0, width=0)


def render_standby_viewport(placeholder, is_warning=False):
    """Render màn hình chờ khi chưa có video được nạp"""
    if is_warning:
        h = """
        <div class="video-standby-viewport" style="border: 1.5px dashed #fca5a5; background: #fef2f2;">
            <div class="standby-icon" style="color: #ef4444;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </div>
            <div class="standby-title" style="color: #dc2626;">CHƯA CÓ VIDEO ĐỂ PHÂN TÍCH</div>
            <div class="standby-desc" style="color: #991b1b;">
                Vui lòng <b>kéo-thả hoặc chọn file video (.mp4, .avi, .mov)</b> ở bảng điều khiển bên trái trước khi bắt đầu!
            </div>
        </div>
        """
    else:
        h = """
        <div class="video-standby-viewport">
            <div class="standby-icon">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                    <line x1="8" y1="21" x2="16" y2="21"/>
                    <line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
            </div>
            <div class="standby-title">CHƯA CÓ VIDEO ĐƯỢC TẢI LÊN</div>
            <div class="standby-desc">
                Vui lòng kéo thả file video hành trình (<b>.mp4</b>, <b>.avi</b>, <b>.mov</b>) vào bảng điều khiển bên trái để bắt đầu phân tích.
            </div>
        </div>
        """
    placeholder.markdown(_c(h), unsafe_allow_html=True)


def render_slim_progress(placeholder, current_frame: int, total_frames: int, fps_val: float = 30.0, completed: bool = False):
    """Render thanh tiến độ mảnh dưới đáy khung video: Tiến độ phân tích: Frame X / Y (Z%) ... 00:04 / 00:11"""
    total_frames = max(1, total_frames)
    pct = 100 if completed else min(100, int((current_frame / total_frames) * 100))
    cur_sec = int(current_frame / max(1.0, fps_val))
    tot_sec = int(total_frames / max(1.0, fps_val))
    time_str = f"{cur_sec // 60:02d}:{cur_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}"

    if completed:
        label_html = f'<span class="slim-progress-label" style="color:#059669;">Hoàn tất phân tích: {total_frames} / {total_frames} frames (100%)</span>'
        fill_style = "width: 100%; background: #10b981;"
    else:
        label_html = f'<span class="slim-progress-label">Tiến độ phân tích: Frame {current_frame} / {total_frames} ({pct}%)</span>'
        fill_style = f"width: {pct}%; background: #2563eb;"

    h = f"""
    <div class="slim-progress-bar">
        {label_html}
        <div class="slim-progress-track">
            <div class="slim-progress-fill" style="{fill_style}"></div>
        </div>
        <span class="slim-progress-meta">{time_str}</span>
    </div>
    """
    placeholder.markdown(_c(h), unsafe_allow_html=True)


def render_cockpit_telemetry(placeholder, speed="--", lead_dist="--", ttc="--", risk="STANDBY",
                             ego_lane="Ego Lane (Làn chính)", signal="Clear",
                             ped_hazard="0 Sidewalk (Safe)", lead_id=1,
                             env_info="Nắng ráo", cut_in_alert=False,
                             is_trip_summary: bool = False, safe_ratio_pct: float = 100.0):
    """
    Render Cụm Đồng Hồ ADAS Dạng Băng Ngang Liền Mạch (Ribbon Dock) ở cột phải
    Khớp 100% với dock-item trong pro_video_change_toggle_ui.png
    Hỗ trợ hiển thị: Tạt đầu (Cut-in), Phanh khẩn cấp (AEB Active), Đèn đỏ/vàng/xanh, Điều kiện môi trường.
    """
    lbl_speed = "AVG SPEED" if is_trip_summary else "CURRENT SPEED"
    tag_speed = "AVERAGE" if is_trip_summary else "NOMINAL"

    lbl_lead = "MIN LEAD GAP" if is_trip_summary else "LEAD VEHICLE GAP"
    if cut_in_alert:
        tag_lead = "⚠️ CUT-IN!"
        tag_lead_color = "#dc2626"
    else:
        tag_lead = "#1 EGO LANE" if lead_id is not None else "CLEAR"
        tag_lead_color = "#0284c7"

    lbl_ttc = "MIN TTC" if is_trip_summary else "TIME TO COLLISION"
    tag_ttc = "THRESHOLD < 3.5S"

    lbl_risk = "TRIP ASSESSMENT" if is_trip_summary else "RISK ASSESSMENT"
    tag_risk = "ADAS FCW / AEB"

    # 1. Tốc độ
    if isinstance(speed, (int, float)):
        speed_val_txt = str(speed)
        speed_pct = min(100, max(10, int((float(speed) / 80.0) * 100)))
    else:
        speed_val_txt = str(speed)
        speed_pct = 0

    # 2. Khoảng cách xe trước
    if isinstance(lead_dist, (int, float)):
        gap_val_txt = str(lead_dist)
        gap_pct = min(100, max(12, int((float(lead_dist) / 45.0) * 100)))
    else:
        gap_val_txt = str(lead_dist)
        gap_pct = 100 if str(lead_dist).lower() in ["clear", "trống"] else 0

    # 3. TTC
    if isinstance(ttc, (int, float)):
        ttc_val_txt = str(ttc)
        ttc_val_f = float(ttc)
        if ttc_val_f < 1.3:
            ttc_color = "#dc2626"
            ttc_bar_color = "#dc2626"
        elif ttc_val_f < 3.0:
            ttc_color = "#d97706"
            ttc_bar_color = "#f59e0b"
        else:
            ttc_color = "#059669"
            ttc_bar_color = "#10b981"
        ttc_pct = min(100, max(15, int((ttc_val_f / 6.0) * 100)))
    else:
        ttc_val_txt = str(ttc)
        if str(ttc).lower() in ["safe", "stable", "an toàn", "ổn định"]:
            ttc_color = "#059669"
            ttc_bar_color = "#10b981"
            ttc_pct = 85
        else:
            ttc_color = "#0f172a"
            ttc_bar_color = "#94a3b8"
            ttc_pct = 0

    # 4. Đánh giá rủi ro
    if is_trip_summary:
        if safe_ratio_pct >= 85.0:
            risk_color = "#059669"
            risk_badge = f"● {safe_ratio_pct:.1f}% SAFE"
            risk_bar_color = "#10b981"
            risk_pct = int(safe_ratio_pct)
        elif safe_ratio_pct >= 60.0:
            risk_color = "#d97706"
            risk_badge = f"● {safe_ratio_pct:.1f}% SAFE"
            risk_bar_color = "#f59e0b"
            risk_pct = int(safe_ratio_pct)
        else:
            risk_color = "#dc2626"
            risk_badge = f"● {safe_ratio_pct:.1f}% SAFE"
            risk_bar_color = "#dc2626"
            risk_pct = int(safe_ratio_pct)
    else:
        r_upper = str(risk).upper()
        if any(k in r_upper for k in ["EMERGENCY", "AEB", "KHAN CAP", "PHANH"]):
            risk_color = "#dc2626"
            risk_badge = "🚨 AEB ACTIVE"
            risk_bar_color = "#dc2626"
            risk_pct = 100
        elif r_upper in ["CAUTION", "CHU Y", "WARN", "WARNING"]:
            risk_color = "#d97706"
            risk_badge = "● CAUTION"
            risk_bar_color = "#f59e0b"
            risk_pct = 68
        elif r_upper in ["DANGER", "NGUY HIEM", "CRITICAL"]:
            risk_color = "#dc2626"
            risk_badge = "● DANGER"
            risk_bar_color = "#dc2626"
            risk_pct = 92
        elif r_upper in ["SAFE", "AN TOAN"]:
            risk_color = "#059669"
            risk_badge = "● SAFE"
            risk_bar_color = "#10b981"
            risk_pct = 25
        else:
            risk_color = "#64748b"
            risk_badge = "○ STANDBY"
            risk_bar_color = "#94a3b8"
            risk_pct = 0

    # Màu sắc đèn tín hiệu
    sig_u = str(signal).upper()
    if any(k in sig_u for k in ["RED", "ĐỎ"]):
        sig_color = "#dc2626"
    elif any(k in sig_u for k in ["YELLOW", "VÀNG", "AMBER"]):
        sig_color = "#d97706"
    elif any(k in sig_u for k in ["GREEN", "XANH"]):
        sig_color = "#059669"
    else:
        sig_color = "#64748b"

    ped_color = "#dc2626" if any(k in str(ped_hazard).upper() for k in ["WARN", "HAZARD", "LÒNG ĐƯỜNG", "ROAD"]) and not str(ped_hazard).startswith("0") else "#059669"
    env_color = "#dc2626" if any(k in str(env_info).upper() for k in ["MỜ", "BLUR", "BÃO", "STORM"]) else ("#d97706" if any(k in str(env_info).upper() for k in ["MƯA", "SƯƠNG", "ĐÊM"]) else "#059669")

    h = f"""
    <div class="cockpit-ribbon-card">
        <div class="metric-dock-grid">
            <div class="dock-item">
                <div class="dock-label">
                    <span>{lbl_speed}</span>
                    <span style="color:#2563eb;">{tag_speed}</span>
                </div>
                <div class="dock-value" style="color:#2563eb;">
                    {speed_val_txt} <span class="dock-unit">km/h</span>
                </div>
                <div class="dock-meter">
                    <div class="meter-fill" style="width: {speed_pct}%; background: #2563eb;"></div>
                </div>
            </div>
            <div class="dock-item">
                <div class="dock-label">
                    <span>{lbl_lead}</span>
                    <span style="color:{tag_lead_color};">{tag_lead}</span>
                </div>
                <div class="dock-value">
                    {gap_val_txt} <span class="dock-unit">m</span>
                </div>
                <div class="dock-meter">
                    <div class="meter-fill" style="width: {gap_pct}%; background: {tag_lead_color};"></div>
                </div>
            </div>
            <div class="dock-item">
                <div class="dock-label">
                    <span>{lbl_ttc}</span>
                    <span style="color:#d97706;">{tag_ttc}</span>
                </div>
                <div class="dock-value" style="color: {ttc_color};">
                    {ttc_val_txt} <span class="dock-unit">s</span>
                </div>
                <div class="dock-meter">
                    <div class="meter-fill" style="width: {ttc_pct}%; background: {ttc_bar_color};"></div>
                </div>
            </div>
            <div class="dock-item">
                <div class="dock-label">
                    <span>{lbl_risk}</span>
                    <span>{tag_risk}</span>
                </div>
                <div class="dock-value" style="color: {risk_color}; font-size: 22px; padding-top: 2px;">
                    {risk_badge}
                </div>
                <div class="dock-meter">
                    <div class="meter-fill" style="width: {risk_pct}%; background: {risk_bar_color};"></div>
                </div>
            </div>
        </div>
        <div class="sub-status-bar">
            <div class="status-item">
                <span>Current Lane:</span> <span class="status-val">{ego_lane}</span>
            </div>
            <div class="status-item">
                <span>Traffic Light:</span> <span class="status-val" style="color: {sig_color};">{signal}</span>
            </div>
            <div class="status-item">
                <span>Pedestrians:</span> <span class="status-val" style="color: {ped_color};">{ped_hazard}</span>
            </div>
            <div class="status-item">
                <span>Environment:</span> <span class="status-val" style="color: {env_color};">{env_info}</span>
            </div>
        </div>
    </div>
    """
    placeholder.markdown(_c(h), unsafe_allow_html=True)


def _highlight_json_line(line: str) -> str:
    """Tô màu cú pháp JSON chuẩn GitHub Light"""
    escaped = html.escape(line)
    escaped = re.sub(r'(&quot;[^&]+?&quot;)(\s*:)', r'<span class="jk">\1</span>\2', escaped)
    escaped = re.sub(r'(:\s*)(&quot;[^&]*?&quot;)', r'\1<span class="js">\2</span>', escaped)
    escaped = re.sub(r'^(\s*)(&quot;[^&]*?&quot;)(,?)$', r'\1<span class="js">\2</span>\3', escaped)
    escaped = re.sub(r'(:\s*)(true|false|null)\b', r'\1<span class="jb">\2</span>', escaped)
    escaped = re.sub(r'(:\s*)(-?\d+(?:\.\d+)?)', r'\1<span class="jn">\2</span>', escaped)
    return escaped


def render_terminal_json(placeholder, json_data: dict, summary_text: str = None, conclusion_info: dict = None):
    """
    Render NHẬT KÝ NGỮ CẢNH THỜI GIAN THỰC ở cột trái
    Có summary-strip viền xanh bên trái + Code box đánh số dòng
    """
    if conclusion_info is not None:
        total_f = conclusion_info.get("total_frames", 0)
        badge_html = f'<span class="card-badge-done">HOÀN TẤT • {total_f} FRAMES</span>'
        eval_label = conclusion_info.get("eval_label", "Đánh giá: An toàn")
        eval_color = conclusion_info.get("eval_color", "#059669")
        summary_html_text = conclusion_info.get("summary_html", "")
        banner_html = (
            f'<div class="summary-strip" style="border-left-color: {eval_color};">'
            f'<div style="font-weight:700; color:{eval_color}; margin-bottom:2px;">KẾT LUẬN NGỮ CẢNH HÀNH TRÌNH • {eval_label}</div>'
            f'<div>{summary_html_text}</div>'
            f'</div>'
        )
    else:
        frame_idx = json_data.get("frame_index") or json_data.get("frame")
        if frame_idx is not None:
            badge_html = f'<span class="frame-pill">FRAME #{frame_idx}</span>'
        else:
            badge_html = f'<span class="frame-pill">{json_data.get("status", "STANDBY")}</span>'

        if summary_text:
            banner_html = f'<div class="summary-strip">{summary_text}</div>'
        else:
            banner_html = ""

    formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
    formatted_json = re.sub(
        r'\[\s+(-?\d+(?:\.\d+)?),\s+(-?\d+(?:\.\d+)?),\s+(-?\d+(?:\.\d+)?),\s+(-?\d+(?:\.\d+)?)\s+\]',
        r'[\1, \2, \3, \4]',
        formatted_json
    )

    lines = formatted_json.splitlines()
    line_nums_html = "<br>".join(str(i) for i in range(1, len(lines) + 1))
    code_lines_html = "<br>".join(
        _highlight_json_line(l).replace(" ", "&nbsp;") for l in lines
    )

    h = f"""
    <div class="context-log-card">
        <div class="card-header-row">
            <span class="section-label" style="margin:0;">NHẬT KÝ NGỮ CẢNH THỜI GIAN THỰC</span>
            {badge_html}
        </div>
        {banner_html}
        <div class="json-inspector-box">
            <div class="json-scroll-viewport">
                <div class="json-line-nums">{line_nums_html}</div>
                <div class="json-code-content">{code_lines_html}</div>
            </div>
        </div>
    </div>
    """
    placeholder.markdown(_c(h), unsafe_allow_html=True)


def inject_bottom_align_script():
    """Tự động đo vị trí đáy của Cụm đồng hồ ADAS bên phải và kéo dài khung Nhật ký JSON bên trái bằng khít 100%"""
    components.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        function syncBottomEdge() {
            const logCard = doc.querySelector('.context-log-card');
            const scrollVp = doc.querySelector('.json-scroll-viewport');
            const ribbonCard = doc.querySelector('.cockpit-ribbon-card');
            if (!logCard || !scrollVp || !ribbonCard) return;

            let parentEl = logCard.parentElement;
            for (let i = 0; i < 4 && parentEl; i++) {
                parentEl.style.overflow = 'visible';
                parentEl = parentEl.parentElement;
            }

            const ribbonRect = ribbonCard.getBoundingClientRect();
            const logRect = logCard.getBoundingClientRect();
            const vpRect = scrollVp.getBoundingClientRect();

            if (ribbonRect.bottom > logRect.top + 100 && vpRect.top >= logRect.top) {
                const targetCardHeight = Math.round(ribbonRect.bottom - logRect.top);
                const usedAboveVp = Math.round(vpRect.top - logRect.top);
                const bottomPad = 14;
                const targetVpHeight = Math.max(140, targetCardHeight - usedAboveVp - bottomPad);

                logCard.style.height = targetCardHeight + 'px';
                logCard.style.maxHeight = targetCardHeight + 'px';
                scrollVp.style.height = targetVpHeight + 'px';
                scrollVp.style.maxHeight = targetVpHeight + 'px';
            }
        }
        syncBottomEdge();
        setInterval(syncBottomEdge, 250);
        window.parent.addEventListener('resize', syncBottomEdge);
    })();
    </script>
    """, height=0, width=0)
