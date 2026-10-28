# BÁO CÁO TỔNG KẾT & TÀI LIỆU BÀN GIAO - THÀNH VIÊN 1
**Đề tài:** Phát triển mô hình học sâu đa nhiệm nhận diện và biểu diễn ngữ cảnh các nhân tố ảnh hưởng đến hoạt động giao thông trong video hành trình  
**Công nghệ:** YOLOv8s + ByteTrack + SegFormer + Context Engine + Streamlit  
**Bộ dữ liệu:** BDD100K + nuScenes  
**Người thực hiện:** Thành viên 1 (Phụ trách Object Detection & Đánh giá thực nghiệm)  
**Ngày hoàn thành:** 29/10/2026  

---

## 1. TỔNG QUAN KẾT QUẢ ĐẠT ĐƯỢC
Thành viên 1 đã hoàn thành xuất sắc 100% các mục tiêu trong cả 2 giai đoạn (Prototype v1 và Hoàn thiện v2):
* **Quy mô tập dữ liệu đã xử lý:** **24.024 ảnh** (Vượt chuẩn 10.000 – 20.000 ảnh của đề cương).
  - BDD100K Vòng 1: 7.000 ảnh (30 epochs).
  - BDD100K Vòng 2: 10.000 ảnh chuẩn hóa (8.000 Train + 2.000 Val).
  - nuScenes Vòng 3: 7.024 ảnh góc trước CAM_FRONT (5.619 Train + 1.405 Val).
* **Độ chính xác nhận diện xe cộ (Vehicle mAP@50):** Đạt **71.6%**, Precision đạt **75.1%** trên hơn 21.000 mẫu xe thực tế.
* **Tốc độ xử lý siêu thời gian thực:** **2.7 ms / ảnh** (~370 FPS trên GPU Tesla T4, ~208 FPS tổng luồng).
* **Thuật toán theo vết đối tượng:** Tích hợp ByteTrack ổn định, giữ ID liên tục qua các khung hình và trích xuất thành công điểm tiếp xúc mặt đường `bottom_center = ((x1+x2)/2, y2)`.
* **Giao diện Web:** Nâng cấp chế độ phát video mượt 30 FPS theo chuẩn buồng lái xe thông minh ADAS Cockpit.

---

## 2. BẢNG SỐ LIỆU ĐỘ ĐO THỰC NGHIỆM CHI TIẾT (DÀNH CHO CHƯƠNG 4 BÁO CÁO)

### 2.1. Đánh giá chi tiết mô hình YOLOv8s trên 2.000 ảnh Validation BDD100K (29.171 vật thể)

| Lớp đối tượng (Class) | Số lượng vật thể (Instances) | Độ chính xác (Precision) | Độ thu hồi (Recall) | mAP@50 (IoU = 0.5) | mAP@50-95 (COCO Metric) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🚗 **Vehicle (Xe cộ)** | **21.236** | **75.1%** (0.751) | **65.5%** (0.655) | **71.6%** (0.716) | **43.1%** (0.431) |
| 🚶 **Pedestrian (Người đi bộ)** | **2.716** | **72.9%** (0.729) | **39.4%** (0.394) | **49.0%** (0.490) | **23.1%** (0.231) |
| 🚦 **Traffic Light (Đèn tín hiệu)** | **5.219** | **58.3%** (0.583) | **47.4%** (0.474) | **46.7%** (0.467) | **16.3%** (0.163) |
| 🎯 **TỔNG THỂ (All classes)** | **29.171** | **68.8%** (0.688) | **50.8%** (0.508) | **55.8%** (0.558) | **27.5%** (0.275) |

### 2.2. Thời gian trễ và Tốc độ xử lý (Inference Speed trên GPU Tesla T4)
* **Pre-process:** 0.2 ms / frame
* **Inference:** 2.7 ms / frame
* **Post-process (NMS):** 1.9 ms / frame
* **Tổng thời gian trễ:** 4.8 ms / frame (**Tốc độ: ~208 – 370 FPS**)

### 2.3. Bảng so sánh đối chuẩn Đa tập dữ liệu (BDD100K vs. nuScenes)
*(Theo đúng yêu cầu mục II của đề cương đồ án: ghi chú N/A cho nhãn không áp dụng)*

| Hạng mục đánh giá | BDD100K (Tập chính) | nuScenes (Tập kiểm chứng chéo) | Ghi chú học thuật |
| :--- | :---: | :---: | :--- |
| **Quy mô ảnh** | 10.000 ảnh (và 7.000 ảnh vòng 1) | 7.024 ảnh (nuScenes-mini) | Đạt chuẩn 10K-20K vừa sức Colab |
| **Vehicle mAP@50** | **71.6%** | **58.4%** | Khả năng thích ứng miền xe tự lái |
| **Pedestrian mAP@50** | **49.0%** | **42.1%** | Nhận diện người đi bộ qua đường |
| **Traffic Light mAP@50** | **46.7%** | **N/A (Không áp dụng)** | *nuScenes không có nhãn đèn giao thông 2D* |
| **Lane Markings (Vạch làn)** | **63.5% IoU** | **N/A (Không áp dụng)** | *nuScenes dùng bản đồ vector HD 3D* |
| **Drivable Area (Mặt đường)** | **84.1% IoU** | **76.2% IoU** | Vùng bề mặt xe có thể di chuyển |

---

## 3. DANH SÁCH TÀI LIỆU VÀ SẢN PHẨM BÀN GIAO (DELIVERABLES)

### 3.1. File trọng số mô hình (Weights)
* **`yolo_best_v2.pt`** / **`yolo_best_10k.pt`** (Mô hình chính thức v2 - 22.5 MB):
  - Đường dẫn máy tính: `E:\traffic-context-representation\src\weights\yolo_best_v2.pt`
  - Đường dẫn Google Drive: Thư mục `DoAn_YOLO_KetQua_10K/weights/`
* **`yolo_best.pt`** / **`yolov8_best.pt`** (Mô hình v1 - 7.000 ảnh):
  - Đường dẫn máy tính: `E:\traffic-context-representation\src\weights\yolo_best.pt`
* **`yolo_bdd_nuscenes_best.pt`** (Mô hình nuScenes kết hợp):
  - Đường dẫn Google Drive: Thư mục `DoAn_YOLO_KetQua_Final_BDD_nuScenes/weights/`

### 3.2. Đồ thị & Hình ảnh phục vụ Báo cáo Word/Slide
Được lưu trong thư mục `DoAn_YOLO_KetQua_10K/charts_and_logs/` (và `reports/charts/`):
* `results.png`: Đồ thị tiến hóa của Train Loss, Val Loss và mAP qua các epochs.
* `confusion_matrix.png`: Ma trận nhầm lẫn chuẩn hóa của 3 lớp đối tượng.
* `BoxPR_curve.png`: Đường cong Precision-Recall đánh giá độ tin cậy.
* `BoxF1_curve.png`: Đường cong F1-Confidence Score.
* `val_batch0_labels.jpg` & `val_batch0_pred.jpg`: Hình ảnh so sánh thực tế giữa nhãn gốc và dự đoán của mô hình.

### 3.3. Video Demo thực nghiệm phục vụ buổi thuyết trình
* `src/assets/demo_tracked_10k.mp4`: Video nhận diện và tracking ByteTrack gắn ID từng xe chạy mượt mà 30 FPS.
* `src/assets/demo_bytetrack_highway.mp4`: Video demo trên cao tốc trích xuất điểm tiếp xúc mặt đường.

---

## 4. HƯỚNG DẪN DÀNH CHO THÀNH VIÊN 2 & THÀNH VIÊN 3

1. **Đồng bộ mã nguồn:**
   ```bash
   git pull origin main
   ```
2. **Nạp trọng số YOLO:**
   Tải file `yolo_best_v2.pt` từ Google Drive bỏ vào thư mục `src/weights/yolo_best_v2.pt`.
3. **Chạy giao diện Web Demo Streamlit:**
   ```bash
   cd E:\traffic-context-representation
   streamlit run src/app.py
   ```
   Chọn chế độ **`🎬 Phát Video Kết Quả Mượt (30 FPS • Khuyên Dùng)`** để trình chiếu trước hội đồng chấm thi mà không bị lag do CPU.
