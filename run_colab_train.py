"""
Script chay tren Google Colab:
- Tu dong tim anh BDD100K tu Kaggle cache
- Lien ket 80.000 anh va chuyen doi nhan chuan 3 lop
- Huan luyen YOLOv8s 30 epochs
- Tu dong luu yolo_best.pt va bieu do vao Google Drive (DoAn_YOLO)
"""

import os
import glob
import shutil
import json
from tqdm import tqdm
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("BƯỚC 1: TÌM KIẾM 80.000 ẢNH BDD100K TRONG CACHE KAGGLEHUB")
    print("=" * 60)

    train_dirs = [p for p in glob.glob("/root/.cache/kagglehub/**/train", recursive=True) if "images" in p]
    val_dirs = [p for p in glob.glob("/root/.cache/kagglehub/**/val", recursive=True) if "images" in p]

    if not train_dirs or not val_dirs:
        raise FileNotFoundError("Không tìm thấy thư mục ảnh trong cache /root/.cache/kagglehub!")

    actual_train = train_dirs[0]
    actual_val = val_dirs[0]
    print(f"Thư mục ảnh Train: {actual_train}")
    print(f"Thư mục ảnh Val  : {actual_val}")

    OUT_DIR = "/content/dataset_yolo"
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

    # Tạo symlink toàn bộ thư mục
    os.symlink(actual_train, f"{OUT_DIR}/images/train")
    os.symlink(actual_val, f"{OUT_DIR}/images/val")
    print("Đã liên kết thư mục ảnh thành công!")

    print("\n" + "=" * 60)
    print("BƯỚC 2: CHUYỂN ĐỔI NHÃN 3 LỚP (Vehicle, Pedestrian, Traffic Light)")
    print("=" * 60)

    json_train = glob.glob("/root/.cache/kagglehub/**/bdd100k_labels_images_train.json", recursive=True)[0]
    json_val = glob.glob("/root/.cache/kagglehub/**/bdd100k_labels_images_val.json", recursive=True)[0]

    MAPPING = {
        "car": 0, "bus": 0, "truck": 0, "train": 0, "trailer": 0, "motorcycle": 0,
        "pedestrian": 1, "person": 1, "rider": 1, "bicycle": 1,
        "traffic light": 2
    }

    def convert_labels(json_file, split):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Đang xử lý nhãn tập {split} ({len(data):,} ảnh)...")
        for item in tqdm(data):
            name = os.path.splitext(item.get("name", ""))[0]
            if not name:
                continue
            lines = []
            for lbl in item.get("labels", []):
                cat = lbl.get("category", "").strip().lower()
                if cat not in MAPPING:
                    continue
                cid = MAPPING[cat]
                b = lbl.get("box2d", None)
                if not b:
                    continue
                x1 = float(b['x1'])
                y1 = float(b['y1'])
                x2 = float(b['x2'])
                y2 = float(b['y2'])
                xc = max(0.0, min(1.0, (x1 + x2) / 2.0 / 1280.0))
                yc = max(0.0, min(1.0, (y1 + y2) / 2.0 / 720.0))
                bw = max(0.0, min(1.0, (x2 - x1) / 1280.0))
                bh = max(0.0, min(1.0, (y2 - y1) / 720.0))
                if bw > 0 and bh > 0:
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            with open(f"{OUT_DIR}/labels/{split}/{name}.txt", 'w', encoding='utf-8') as f_out:
                f_out.writelines(lines)

    convert_labels(json_train, "train")
    convert_labels(json_val, "val")

    # Tạo data.yaml
    yaml_content = f"""path: {OUT_DIR}
train: images/train
val: images/val
test: images/val

nc: 3
names:
  0: Vehicle
  1: Pedestrian
  2: Traffic Light
"""
    with open("/content/data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print("\n" + "=" * 60)
    print("BƯỚC 3: BẮT ĐẦU HUẤN LUYỆN YOLOv8s TRÊN GPU TESLA T4 (30 EPOCHS)")
    print("=" * 60)

    model = YOLO("yolov8s.pt")
    results = model.train(
        data="/content/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        workers=4,
        name="yolov8s_traffic_best",
        save=True
    )

    print("\n" + "=" * 60)
    print("BƯỚC 4: TỰ ĐỘNG LƯU TRỌN BỘ KẾT QUẢ VÀO GOOGLE DRIVE (DoAn_YOLO)")
    print("=" * 60)

    drive_save_dir = "/content/drive/MyDrive/DoAn_YOLO"
    os.makedirs(drive_save_dir, exist_ok=True)

    best_weight = "runs/detect/yolov8s_traffic_best/weights/best.pt"
    if os.path.exists(best_weight):
        shutil.copy2(best_weight, f"{drive_save_dir}/yolo_best.pt")
        print(f"Đã lưu: {drive_save_dir}/yolo_best.pt")

    shutil.copytree("runs/detect/yolov8s_traffic_best", f"{drive_save_dir}/ket_qua_bao_cao", dirs_exist_ok=True)
    print(f"Đã lưu toàn bộ biểu đồ báo cáo: {drive_save_dir}/ket_qua_bao_cao/")
    print("🎉 TẤT CẢ CÔNG VIỆC CỦA THÀNH VIÊN 1 ĐÃ HOÀN TẤT XUẤT SẮC!")

if __name__ == "__main__":
    main()
