"""
Chuyển đổi nhãn BDD100k từ định dạng JSON sang file txt theo chuẩn YOLOv12
Phục vụ tiền xử lý dữ liệu huấn luyện phối hợp (Joint-Training) trên Google Colab.
"""
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from src.utils.label_mapper import map_bdd_detection
except ImportError:
    from utils.label_mapper import map_bdd_detection


def convert_bdd100k_to_yolo(box2d, img_width, img_height):
    """
    Chuyển đổi bounding box từ định dạng BDD100k sang định dạng YOLO
    :param box2d: dict chứa thông tin bounding box theo định dạng BDD100k
    :param img_width: chiều rộng ảnh
    :param img_height: chiều cao ảnh
    :return: list [x_center, y_center, width, height] theo chuẩn YOLO
    """
    x1 = float(box2d['x1'])
    y1 = float(box2d['y1'])
    x2 = float(box2d['x2'])
    y2 = float(box2d['y2'])

    # Tính toán tọa độ trung tâm và kích thước của bounding box theo chuẩn YOLO
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Giới hạn trong khoảng [0, 1] để tránh lỗi khi bounding box nằm ngoài ảnh
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def process_bdd_labels(json_path, output_dir, img_width=1280, img_height=720):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Đang đọc nhãn từ file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Đang chuyển đổi {len(data)} nhãn sang định dạng YOLO...")
    converted_count = 0
    box_count = 0
    for item in tqdm(data, desc="Converting BDD100K"):
        img_name = item.get("name", "")
        if not img_name:
            continue

        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(output_dir, f"{base_name}.txt")

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

                x_center, y_center, width, height = convert_bdd100k_to_yolo(box2d, img_width, img_height)

                if width <= 0 or height <= 0:
                    continue

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                box_count += 1

        with open(label_file, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        converted_count += 1
    print(f"Hoàn tất chuyển đổi. Đã xử lý {converted_count} nhãn và tạo {box_count} bounding box.")


if __name__ == "__main__":
    # Script này dùng để tiền xử lý tập nhãn BDD100K JSON sang YOLO txt trên Google Colab
    json_val = "/content/bdd100k_raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    if not os.path.exists(json_val):
        json_val = "/content/bdd100k_raw/bdd100k_labels_release/bdd100k_labels_images_val.json"

    if not os.path.exists(json_val):
        print("[THONG BAO] Script nay duoc thiet ke de tien xu ly du lieu tho BDD100K tren Google Colab.")
        print(f"           Khong tim thay tap nhan JSON tai: /content/bdd100k_raw/...")
        print("           Du an hien tai da co san trong so trong 'weights/', ban co the chay truc tiep Web Demo bang: .venv\\Scripts\\streamlit run src/app.py")
    else:
        out_val = "/content/dataset_processed/labels/val"
        os.makedirs(out_val, exist_ok=True)
        print(f"Đang xử lý tập val từ: {json_val}")
        process_bdd_labels(json_val, out_val)

        json_train = "/content/bdd100k_raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
        if not os.path.exists(json_train):
            json_train = "/content/bdd100k_raw/bdd100k_labels_release/bdd100k_labels_images_train.json"

        if os.path.exists(json_train):
            out_train = "/content/dataset_processed/labels/train"
            os.makedirs(out_train, exist_ok=True)
            print(f"Đang xử lý tập train từ: {json_train}")
            process_bdd_labels(json_train, out_train)

        print("Chuyển đổi toàn bộ nhãn BDD100k hoàn tất!")
