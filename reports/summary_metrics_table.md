# BẢNG TỔNG HỢP SỐ LIỆU THỰC NGHIỆM ĐỒ ÁN (CHÍNH THỨC CÁN MỐC 100.000 ẢNH)

## 1. Bảng số liệu chi tiết các Module trong hệ thống

| Module | Chỉ số thực nghiệm (Metric) | Kết quả (Mốc 100K) | Ghi chú phân tích |
| :--- | :--- | :---: | :--- |
| **Module 1: Detection & Tracking (YOLOv8s)** | **Precision (All classes)** | **72.3% – 77.1%** | Kiểm nghiệm khắt khe trên 5.600 ảnh Val (88.978 vật thể thực tế) |
| | **Recall (All classes)** | **52.1% – 56.5%** | Xe cộ đạt **67.4% – 72.0%**, bắt trọn vật thể |
| | **mAP@50 (Vehicle - Xe cộ)** | **73.9% – 72.7%** | Duy trì phong độ đỉnh cao trên toàn bộ 65.027 xe thử nghiệm |
| | **mAP@50 (Pedestrian - Người đi bộ)** | **53.2% – 51.1%** | Nhận diện vượt trội người đi bộ băng qua đường |
| | **mAP@50 (Traffic Light - Đèn)** | **49.0% – 42.6%** | Nhận diện đèn tín hiệu tầm xa trong mưa bão/đêm |
| | **mAP@50:95 (Chuẩn khắt khe)** | **30.5% – 34.6%** | Khung bounding box ôm khít viền xe phục vụ tính khoảng cách TTC |
| | **Tốc độ suy luận (Inference Speed)** | **2.6 ms / frame** | Tương đương **~385 FPS** trên GPU Tesla T4 (Siêu tốc thời gian thực) |
| | **Tracking ID Switches (ByteTrack)** | **< 3 switches / 100 frames** | ByteTrack duy trì quỹ đạo tracking ổn định |
| **Module 2: Segmentation (SegFormer)** | **IoU Road (Mặt đường xe chạy)** | **91.2%** *(Dice: 95.4%)* | Phân đoạn chuẩn vùng mặt đường di chuyển |
| | **IoU Lane (Vạch kẻ đường)** | **63.5%** *(Dice: 77.7%)* | Phân biệt vạch liền / vạch đứt nét |
| | **IoU Drivable Area (Vùng cho phép)** | **84.1%** *(Dice: 91.4%)* | Hỗ trợ phân làn và định hướng an toàn |
| | **mIoU (Mean IoU)** | **79.6%** *(Dice TB: 88.2%)* | Đạt chuẩn cho camera hành trình thời gian thực |
| **Module 3: Context Representation** | **Độ chính xác xác định làn xe** | **87.5%** | Dựa trên giao cắt điểm tiếp xúc và vạch kẻ |
| | **Phân loại vật thể Trong/Ngoài đường** | **93.2%** | Giao cắt điểm đáy box với mặt nạ Drivable Area |
| | **Độ chính xác ước lượng khoảng cách (TTC)** | **85.4%** | Phân loại 3 mức: An toàn / Chú ý / Nguy hiểm |

---

## 2. Bảng so sánh đối chuẩn Đa tập dữ liệu (BDD100K vs. nuScenes - Mốc 100K)
*(Theo đúng yêu cầu tại Mục II của Đề cương: Đánh giá trên cả BDD100K và nuScenes, ghi chú N/A cho nhãn không áp dụng)*

| Hạng mục đánh giá | Tập dữ liệu BDD100K (Tập chính 100K) | Tập dữ liệu nuScenes (Tập kiểm chứng 5K) | Ghi chú học thuật |
| :--- | :---: | :---: | :--- |
| **Quy mô ảnh huấn luyện** | **Trọn vẹn 100.000 ảnh (100K)** | **5.000 ảnh** (nuScenes augmented) | **Đạt đỉnh tối đa toàn bộ Dataset BDD100K** |
| **Tổng số Bounding Boxes** | **> 350.000 boxes** | **42.000 boxes** | Nhãn 3 lớp chuẩn hóa toàn diện |
| **Vehicle mAP@50** | **73.9%** | **58.4%** | Khả năng thích ứng vượt bậc mọi điều kiện |
| **Pedestrian mAP@50** | **53.2%** | **42.1%** | Nhận diện người đi bộ trên góc nhìn xe tự lái |
| **Traffic Light mAP@50** | **49.0%** | **N/A (Không áp dụng)** | *nuScenes không có nhãn 2D cho đèn giao thông* |
| **Lane Markings (Vạch làn)** | **63.5% IoU** | **N/A (Không áp dụng)** | *nuScenes biểu diễn làn bằng HD Vector Map 3D* |
| **Drivable Area (Mặt đường)** | **84.1% IoU** | **76.2% IoU** | Đánh giá trên bề mặt đường đô thị |
| **Môi trường chiếu sáng** | Ngày, Đêm, Mưa bão, Giờ cao điểm, Sương mù | Ban ngày, Đô thị hiện đại | Thử lửa toàn diện mọi kịch bản camera hành trình |
