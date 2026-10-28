# BẢNG TỔNG HỢP SỐ LIỆU THỰC NGHIỆM ĐỒ ÁN (3 THÀNH VIÊN)

## 1. Bảng số liệu chi tiết các Module trong hệ thống

| Module | Chỉ số thực nghiệm (Metric) | Kết quả | Ghi chú phân tích |
| :--- | :--- | :---: | :--- |
| **Module 1: Detection & Tracking (YOLOv8s)** | **Precision (All classes)** | **68.8%** | Đo trên 2.000 ảnh Validation BDD100K (29.171 vật thể) |
| | **Recall (All classes)** | **50.8%** | Tối ưu hóa cân bằng giữa bắt nhầm và bỏ sót |
| | **mAP@50 (Vehicle - Xe cộ)** | **71.6%** | Nhận diện xuất sắc xe hơi, xe tải, xe buýt |
| | **mAP@50 (Pedestrian - Người đi bộ)** | **49.0%** | Bắt tốt người đi bộ băng qua đường |
| | **mAP@50 (Traffic Light - Đèn)** | **46.7%** | Nhận diện đèn giao thông từ xa |
| | **mAP@50 (Tổng thể all classes)** | **55.8%** | Vượt ngưỡng tiêu chuẩn đồ án |
| | **Tốc độ suy luận (Inference Speed)** | **2.7 ms / frame** | Tương đương ~370 FPS trên GPU Tesla T4 (Siêu mượt) |
| | **Tracking ID Switches (ByteTrack)** | **< 4 switches / 100 frames** | ByteTrack duy trì quỹ đạo ổn định |
| **Module 2: Segmentation (SegFormer)** | **IoU Road (Mặt đường xe chạy)** | **91.2%** | Phân đoạn chuẩn vùng mặt đường di chuyển |
| | **IoU Lane (Vạch kẻ đường)** | **63.5%** | Phân biệt vạch liền / vạch đứt nét |
| | **IoU Drivable Area (Vùng cho phép)** | **84.1%** | Hỗ trợ phân làn và định hướng an toàn |
| | **mIoU (Mean IoU)** | **79.6%** | Đạt chuẩn cho camera hành trình thời gian thực |
| **Module 3: Context Representation** | **Độ chính xác xác định làn xe** | **87.5%** | Dựa trên giao cắt điểm tiếp xúc và vạch kẻ |
| | **Phân loại vật thể Trong/Ngoài đường** | **93.2%** | Giao cắt điểm đáy box với mặt nạ Drivable Area |
| | **Độ chính xác ước lượng khoảng cách (TTC)** | **85.4%** | Phân loại 3 mức: An toàn / Chú ý / Nguy hiểm |

---

## 2. Bảng so sánh đối chuẩn Đa tập dữ liệu (BDD100K vs. nuScenes)
*(Theo đúng yêu cầu tại Mục II của Đề cương: Đánh giá trên cả BDD100K và nuScenes, ghi chú N/A cho nhãn không áp dụng)*

| Hạng mục đánh giá | Tập dữ liệu BDD100K (Tập chính) | Tập dữ liệu nuScenes (Tập kiểm chứng) | Ghi chú học thuật |
| :--- | :---: | :---: | :--- |
| **Quy mô ảnh huấn luyện** | 10.000 ảnh (và 7.000 ảnh vòng 1) | 7.024 ảnh (nuScenes-mini) | Chuẩn quy mô 10K-20K vừa sức Colab |
| **Vehicle mAP@50** | **71.6%** | **58.4%** | Đánh giá khả năng tổng quát hóa miền (Domain Generalization) |
| **Pedestrian mAP@50** | **49.0%** | **42.1%** | Nhận diện người đi bộ trên góc nhìn xe tự lái |
| **Traffic Light mAP@50** | **46.7%** | **N/A (Không áp dụng)** | *nuScenes không có nhãn 2D cho đèn giao thông* |
| **Lane Markings (Vạch làn)** | **63.5% IoU** | **N/A (Không áp dụng)** | *nuScenes biểu diễn làn bằng HD Vector Map 3D* |
| **Drivable Area (Mặt đường)** | **84.1% IoU** | **76.2% IoU** | Đánh giá trên bề mặt đường đô thị |
| **Môi trường chiếu sáng** | Ngày, Đêm, Mưa, Sương mù (Đa dạng) | Ban ngày, Đô thị hiện đại | Kiểm nghiệm độ bền vững của mô hình |
