from ultralytics import YOLO
import cv2
import os

def main():
    # 1. Load model YOLOv8 (đảm bảo file weights yolov8_best.pt đã được đặt trong thư mục dự án)
    weight_path = "yolov8_best.pt"  # Hoặc đường dẫn trỏ tới file weights của bạn
    if not os.path.exists(weight_path):
        print(f"❌ Không tìm thấy file trọng số tại: {weight_path}. Vui lòng đưa file weight vào thư mục dự án!")
        return

    print(f"🚀 Đang tải mô hình từ: {weight_path}...")
    model = YOLO(weight_path)

    # 2. Đường dẫn đến video đầu vào cần test tracking
    # (Bạn có thể bỏ một video ngắn quay giao thông vào thư mục dự án và đổi tên ở đây)
    video_path = "test_traffic.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠️ Không tìm thấy video '{video_path}'. Hãy chuẩn bị một file video mẫu để chạy thử nhé!")
        return

    print("🔄 Bắt đầu chạy mô hình với thuật toán ByteTrack...")

    # 3. Chạy model.track() tích hợp sẵn ByteTrack
    results = model.track(
        source=video_path,
        conf=0.3,              # Ngưỡng độ tin cậy
        iou=0.5,               # Ngưỡng IOU cho NMS
        tracker="bytetrack.yaml", # Cấu hình thuật toán ByteTrack
        persist=True,          # Giữ ID liên tục qua các khung hình
        save=True,             # Tự động lưu video kết quả
        device="cpu"           # Đổi thành "0" nếu máy tính của bạn có card đồ họa NVIDIA rời
    )

    print("🎉 Hoàn thành! Video kết quả tracking đã được lưu trong thư mục 'runs/detect/predict...'")

if __name__ == "__main__":
    main()