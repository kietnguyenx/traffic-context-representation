import streamlit as st
import tempfile
import cv2
import json
from pipeline import DrivingPerceptionPipeline

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
        # Khởi tạo Pipeline (sau này truyền weights vào: DrivingPerceptionPipeline(yolo_weights="...", segformer_weights="..."))
        pipeline = DrivingPerceptionPipeline(fps=30)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 384))
            
            # Chạy toàn bộ luồng qua pipeline
            annotated_frame, context_log = pipeline.run_inference(
                frame=frame,
                frame_idx=frame_idx,
                weather=weather_condition
            )
            
            # Hiển thị kết quả lên giao diện web
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            json_placeholder.code(json.dumps(context_log, indent=2, ensure_ascii=False), language="json")
            
            frame_idx += 1
            
        cap.release()