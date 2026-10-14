import streamlit as st
import tempfile
import cv2
import json
from context_engine import TrafficContextEngine

st.set_page_config(page_title="Traffic Context Representation", layout="wide")
st.title("🚗 Hệ Thống Nhận Diện & Biểu Diễn Ngữ Cảnh Giao Thông")

st.sidebar.header("Cấu hình đầu vào")
dataset_choice = st.sidebar.selectbox("Nguồn chuẩn hóa", ["BDD100K", "nuScenes"])
weather_condition = st.sidebar.selectbox("Thời tiết môi trường", ["Nang rao", "Mua", "Ban dem", "Suong mu"])
uploaded_file = st.sidebar.file_uploader("Tải lên video hành trình", type=["mp4", "avi", "mov"])

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Visualized Video Stream")
    video_placeholder = st.empty()

with col2:
    st.subheader("Real-time Context JSON Log")
    json_placeholder = st.empty()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    start_btn = st.sidebar.button("Bắt đầu phân tích")
    if start_btn:
        engine = TrafficContextEngine()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 384))
            
            # Dữ liệu giả lập mẫu để kiểm thử logic trước khi nạp model hoàn chỉnh
            sample_detections = [
                {"id": 1, "class": "Vehicle", "bbox": [260, 200, 380, 310]}
            ]
            
            # Suy luận ngữ cảnh
            context_log = engine.process_frame(
                detections=sample_detections,
                masks={},
                weather_label=weather_condition,
                frame_idx=frame_idx,
                frame_shape=frame.shape[:2]
            )
            
            # Hiển thị cảnh báo lên frame
            y_offset = 30
            for warn in context_log["warnings"]:
                cv2.putText(frame, warn, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            json_placeholder.code(json.dumps(context_log, indent=2, ensure_ascii=False), language="json")
            
            frame_idx += 1
            
        cap.release()