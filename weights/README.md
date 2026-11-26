# Pre-trained Model Weights

Thư mục lưu trữ các tệp trọng số mô hình học sâu phục vụ hệ thống nhận thức đa nhiệm (Multi-task Perception):

| Mô hình | Tệp trọng số | Kiến trúc | Tác vụ & Lớp đối tượng |
| :--- | :--- | :--- | :--- |
| **YOLOv12** | `yolo_best.pt` | YOLOv12-L (Area Attention) + ByteTrack | Phát hiện & bám vết 3 lớp: `Vehicle`, `Pedestrian`, `Traffic Light` |
| **SegFormer** | `segformer_best.pth` | MiT-B2 Transformer | Phân đoạn ngữ nghĩa mặt đường: `Drivable Area`, `Lane Markings` |

---

### Hướng Dẫn Sử Dụng (Quickstart)

1. Tải hoặc đặt 2 tệp trọng số đã huấn luyện vào thư mục này:
   - `weights/yolo_best.pt`
   - `weights/segformer_best.pth`
2. Hệ thống [`DrivingPerceptionPipeline`](../src/pipeline.py) sẽ tự động nạp các tệp này khi khởi động ứng dụng Web Dashboard.
3. Nếu chưa có trọng số thực tế, hệ thống hỗ trợ cơ chế tự sinh dữ liệu giả lập (**Synthetic Fallback**) để kiểm thử luồng giao diện mà không bị dừng đột ngột.
