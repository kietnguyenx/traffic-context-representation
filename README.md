# Traffic Context Representation System

Hệ thống nhận thức đa nhiệm (Multi-task Perception) và biểu diễn ngữ cảnh giao thông thời gian thực trên video hành trình, phục vụ nghiên cứu Hệ thống Hỗ trợ Lái xe Nâng cao (ADAS).

Dự án: *"Phát triển mô hình học sâu đa nhiệm nhận diện và biểu diễn ngữ cảnh các nhân tố ảnh hưởng đến hoạt động giao thông trong video hành trình"*.

---

## 1. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống xử lý tuần tự trên từng khung hình video theo luồng sau:

```
Raw Video Frame (Dashcam)
       │
       ├──► [YOLOv8 + ByteTrack]  ────► Bounding Boxes, Class ID, Track ID
       │
       ├──► [SegFormer]            ────► Drivable Area Mask, Lane Markings Mask
       │
       ▼
[Traffic Context Engine]          ────► Làn xe, Cự ly, TTC, Phân cấp rủi ro
       │
       ▼
[ADAS Cockpit Visualizer]         ────► Streamlit Dashboard / Rendered MP4 Video
```

- **Object Detection & Tracking:** YOLOv8 phát hiện 3 lớp đối tượng (`Vehicle`, `Pedestrian`, `Traffic Light`); ByteTrack duy trì định danh (ID) vật thể qua chuỗi khung hình.
- **Semantic Segmentation:** SegFormer phân đoạn điểm ảnh cho lòng đường (`Drivable Area`) và vạch phân làn (`Lane Markings`).
- **Traffic Context Representation Engine:** Module suy luận không gian xác định vị trí làn xe, kiểm tra vùng an toàn, ước lượng khoảng cách tương đối, tính toán thời gian tới va chạm (Time-To-Collision - TTC) và kích hoạt cảnh báo nguy cơ (FCW, PCW).

---

## 2. Cấu Trúc Thư Mục (Repository Structure)

```
traffic-context-representation/
├── Dockerfile                      # Cấu hình đóng gói container ứng dụng
├── docker-compose.yml              # Cấu hình điều phối dịch vụ Docker
├── README.md                       # Tài liệu kỹ thuật dự án
├── notebooks/                      # Jupyter notebooks huấn luyện và tiền xử lý
│   ├── 01_download_raw_data.ipynb
│   └── 02_train_yolo.ipynb
└── src/                            # Mã nguồn chính của hệ thống
    ├── app.py                      # Ứng dụng Web Dashboard (Streamlit)
    ├── pipeline.py                 # Điều phối luồng nhận thức toàn diện
    ├── context_engine.py           # Module suy luận ngữ cảnh giao thông
    ├── evaluate_context.py         # Kiểm chuẩn định lượng và thí nghiệm bóc tách (Ablation)
    ├── test_context_engine.py      # Unit test kiểm thử độc lập module ngữ cảnh
    ├── export_demo_video.py        # Script kết xuất video MP4 gắn telemetry HUD
    ├── requirements.txt            # Danh sách thư viện phụ thuộc
    ├── context_evaluation_results.json # Kết quả kiểm chuẩn định lượng
    ├── assets/                     # Dữ liệu hình ảnh và video kết quả
    ├── configs/                    # Cấu hình huấn luyện mô hình (data.yaml)
    ├── datasets/                   # Scripts tiền xử lý và ánh xạ nhãn BDD100K/nuScenes
    └── utils/                      # Tiện ích bổ trợ (sinh kịch bản, ánh xạ nhãn, vẽ)
```

---

## 3. Cài Đặt Môi Trường (Installation)

Yêu cầu môi trường Python >= 3.9 (khuyến nghị Python 3.10 hoặc 3.11).

```bash
# Khởi tạo và kích hoạt virtual environment
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate
# Kích hoạt trên Linux / macOS:
# source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r src/requirements.txt
```

---

## 4. Hướng Dẫn Vận Hành (Execution Guide)

### 4.1. Khởi chạy Web Dashboard (Streamlit)
```bash
streamlit run src/app.py --server.port 8501
```
- Truy cập giao diện tại: `http://localhost:8501`
- Hỗ trợ tải trực tiếp video định dạng `.mp4`, `.avi`, `.mov` hoặc chạy kịch bản mô phỏng có sẵn.

### 4.2. Kiểm thử độc lập (Unit Test)
Kiểm tra nhanh tính đúng đắn của logic tính toán TTC, phân làn và kiểm tra vùng đường:
```bash
python src/test_context_engine.py
```

### 4.3. Đánh giá định lượng & Thí nghiệm bóc tách (Evaluation & Ablation)
Đo lường độ chính xác trên 60 frame chuẩn và kiểm tra độ bền vững khi mặt nạ phân đoạn bị nhiễu:
```bash
python src/evaluate_context.py
```
Kết quả đo đạc được xuất ra console và lưu tại `src/context_evaluation_results.json`.

### 4.4. Kết xuất Video Demo mẫu (.mp4)
Render video ngoại tuyến kèm đầy đủ trực quan hóa ADAS HUD và lưu vào file MP4:
```bash
python src/export_demo_video.py --frames 120 --fps 30 --output src/assets/demo_adas_cockpit.mp4
```

### 4.5. Triển khai bằng Docker
```bash
docker compose up --build
```

---

## 5. Đặc Tả Dữ Liệu Ngữ Cảnh Đầu Ra (Output Schema)

Dữ liệu trả về từ `TrafficContextEngine.process_frame()` tuân theo cấu trúc JSON chuẩn hóa:

```json
{
  "frame": 0,
  "weather": "Nang rao",
  "summary": {
    "total_vehicles": 1,
    "total_pedestrians": 0,
    "pedestrians_on_road": 0,
    "traffic_light_ahead": true,
    "traffic_light_distance": "Gan",
    "risk_level": "Nguy hiem",
    "active_warnings_count": 1,
    "fps": 30.0
  },
  "warnings": [
    "CANH BAO VA CHAM (FCW): Xe #1 nguy co cao (TTC: 1.4s, Lan: Lan giua)"
  ],
  "objects": [
    {
      "id": 1,
      "class": "Vehicle",
      "bbox": [250, 220, 390, 350],
      "lane_position": "Lan giua",
      "distance": "Gan",
      "in_drivable": true,
      "ttc": 1.4,
      "risk_level": "Cao",
      "description": "[Vehicle #1] | Lan: Lan giua | K/c: Gan | Trong duong: Co | TTC: 1.4s"
    }
  ]
}
```

