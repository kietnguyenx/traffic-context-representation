"""
Script tự động chuẩn bị TOÀN BỘ dữ liệu BDD100K từ E:\archive.zip cho môi trường VS Code Local.
- Giải nén và xử lý ĐẦY ĐỦ: 70.000 ảnh Train và 10.000 ảnh Val
- Chuyển đổi nhãn sang chuẩn YOLO (0: Vehicle, 1: Pedestrian, 2: Traffic Light)
- Tự động bỏ qua ảnh đã giải nén nếu chạy lại (Resume an toàn)
- Tạo file cấu hình data_local.yaml
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.label_mapper import map_bdd_detection

ARCHIVE_PATH = r"E:\archive.zip"
OUT_BASE = r"E:\dataset_processed"

# None nghĩa là lấy ĐẦY ĐỦ toàn bộ tập dữ liệu (70.000 train + 10.000 val)
NUM_TRAIN = None  
NUM_VAL = None    

def convert_box(box2d, w=1280, h=720):
    x1, y1, x2, y2 = float(box2d['x1']), float(box2d['y1']), float(box2d['x2']), float(box2d['y2'])
    xc = max(0.0, min(1.0, (x1 + x2) / 2.0 / w))
    yc = max(0.0, min(1.0, (y1 + y2) / 2.0 / h))
    bw = max(0.0, min(1.0, (x2 - x1) / w))
    bh = max(0.0, min(1.0, (y2 - y1) / h))
    return xc, yc, bw, bh

def run_preparation():
    if not os.path.exists(ARCHIVE_PATH):
        print(f"❌ Không tìm thấy file {ARCHIVE_PATH}!")
        return

    print("=" * 65)
    print("🚀 CHUẨN BỊ ĐẦY ĐỦ TẬP DỮ LIỆU BDD100K (70.000 TRAIN + 10.000 VAL)")
    print(f"📁 Thư mục xuất dữ liệu: {OUT_BASE}")
    print("=" * 65)
    
    # Tạo cấu trúc thư mục
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUT_BASE, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUT_BASE, 'labels', split), exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as z:
        print("📖 Đang đọc danh sách nhãn JSON từ archive.zip...")
        train_json_name = "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
        val_json_name = "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

        with z.open(train_json_name) as f:
            train_data = json.load(f)
        with z.open(val_json_name) as f:
            val_data = json.load(f)

        train_samples = train_data if NUM_TRAIN is None else train_data[:NUM_TRAIN]
        val_samples = val_data if NUM_VAL is None else val_data[:NUM_VAL]

        print(f"📊 Số lượng ảnh sẽ xử lý: {len(train_samples):,} ảnh Train và {len(val_samples):,} ảnh Val")

        # 1. Xử lý tập TRAIN (70.000 ảnh)
        print(f"\n📦 [1/2] Đang trích xuất ảnh & chuyển đổi nhãn tập TRAIN ({len(train_samples):,} ảnh)...")
        process_split(z, train_samples, 'train')

        # 2. Xử lý tập VAL (10.000 ảnh)
        print(f"\n📦 [2/2] Đang trích xuất ảnh & chuyển đổi nhãn tập VAL ({len(val_samples):,} ảnh)...")
        process_split(z, val_samples, 'val')

    # 3. Tạo file cấu hình data_local.yaml
    yaml_path = project_root / "configs" / "data_local.yaml"
    out_base_clean = OUT_BASE.replace("\\", "/")
    yaml_content = f"""path: {out_base_clean}
train: images/train
val: images/val
test: images/val

nc: 3
names:
  0: Vehicle
  1: Pedestrian
  2: Traffic Light
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())

    print(f"\n✅ Đã cập nhật file cấu hình: {yaml_path}")
    print(f"🎉 Hoàn tất chuẩn bị ĐẦY ĐỦ 80.000 ảnh tại: {OUT_BASE}")

def process_split(z, samples, split_name):
    img_dir = os.path.join(OUT_BASE, 'images', split_name)
    lbl_dir = os.path.join(OUT_BASE, 'labels', split_name)
    zip_img_prefix = f"bdd100k/bdd100k/images/100k/{split_name}/"

    for item in tqdm(samples, desc=f"Split {split_name}", unit="ảnh"):
        img_name = item.get("name", "")
        if not img_name:
            continue

        base_name = os.path.splitext(img_name)[0]
        zip_img_path = zip_img_prefix + img_name

        # 1. Trích xuất ảnh nếu chưa có
        target_img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(target_img_path):
            try:
                img_data = z.read(zip_img_path)
                with open(target_img_path, 'wb') as f_out:
                    f_out.write(img_data)
            except KeyError:
                continue

        # 2. Chuyển đổi nhãn sang YOLO txt nếu chưa có
        label_file = os.path.join(lbl_dir, f"{base_name}.txt")
        if not os.path.exists(label_file):
            yolo_lines = []
            labels = item.get("labels", [])
            if labels:
                for lbl in labels:
                    category = lbl.get("category", "")
                    box2d = lbl.get("box2d", None)
                    if box2d is None:
                        continue

                    class_id = map_bdd_detection(category)
                    if class_id is None:
                        continue

                    xc, yc, bw, bh = convert_box(box2d)
                    if bw <= 0 or bh <= 0:
                        continue

                    yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(yolo_lines)

if __name__ == "__main__":
    run_preparation()
