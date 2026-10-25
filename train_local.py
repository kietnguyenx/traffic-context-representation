"""
Script huấn luyện YOLOv8 trực tiếp trên VS Code bằng GPU RTX 3050 cho toàn bộ tập dữ liệu (70.000 ảnh).
Tác giả: Thành viên 1
"""

import os
import torch
from ultralytics import YOLO

def main():
    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    
    print("=" * 65)
    print("🚀 HUẤN LUYỆN YOLOv8 TRÊN MÁY TÍNH (VS CODE LOCAL)")
    print(f"🖥️ Thiết bị sử dụng: {'NVIDIA GPU (CUDA)' if has_cuda else 'CPU'}")
    if has_cuda:
        print(f"🎮 Card đồ họa: {torch.cuda.get_device_name(0)}")
        print(f"💾 Dung lượng VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    else:
        print("⚠️ Chú ý: Đang chạy bằng CPU. Hãy cài đặt PyTorch CUDA để chạy trên card RTX 3050.")
    print("=" * 65)

    yaml_config = "src/configs/data_local.yaml"
    if not os.path.exists(yaml_config):
        print(f"❌ Chưa tìm thấy {yaml_config}! Hãy chạy script prepare_local_data.py trước.")
        return

    # Khởi tạo mô hình YOLOv8s
    print("\n📦 Đang nạp mô hình yolov8s.pt...")
    model = YOLO("src/weights/yolov8_best.pt" if os.path.exists("src/weights/yolov8_best.pt") else "yolov8s.pt")

    # Huấn luyện mô hình (amp=True bật Tensor Cores trên RTX 3050 để tăng tốc và tiết kiệm VRAM)
    print("\n🔥 Bắt đầu huấn luyện...")
    results = model.train(
        data=yaml_config,
        epochs=30,
        imgsz=640,
        batch=16 if has_cuda else 4,
        device=device,
        workers=4 if has_cuda else 2,
        amp=True,
        name="yolov8s_traffic_full"
    )

    print("\n🎉 Huấn luyện hoàn tất!")
    print("📁 Kết quả lưu tại: runs/detect/yolov8s_traffic_full")

if __name__ == "__main__":
    main()
