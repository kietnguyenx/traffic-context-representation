"""
Module trích xuất dữ liệu ảnh và chuyển đổi nhãn từ nuScenes-mini sang định dạng chuẩn YOLOv12 (txt).
Phục vụ huấn luyện phối hợp (Joint-Training) giữa BDD100K và nuScenes.

Quy ước nhãn mục tiêu cho YOLO:
  0: Vehicle (Ô tô, xe tải, xe buýt,...)
  1: Pedestrian (Người đi bộ)
  2: Traffic Light (Đèn giao thông - nuScenes không có, chỉ có ở BDD100K)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
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
    from src.utils.label_mapper import map_nuscenes_detection
except ImportError:
    from utils.label_mapper import map_nuscenes_detection


def convert_nuscenes_to_yolo(
    dataroot="/content/bdd100k_raw/nuscenes-mini",
    version="v1.0-mini",
    output_img_dir="/content/dataset_processed/images/train",
    output_lbl_dir="/content/dataset_processed/labels/train",
    prefix="nusc_"
):
    """
    Trích xuất ảnh camera trước (CAM_FRONT) và tạo file nhãn YOLO .txt tương ứng.
    (Chỉ sử dụng khi tiền xử lý dữ liệu thô trên Google Colab)
    """
    # Khởi tạo nạp động để tránh cảnh báo thiếu thư viện trên môi trường máy cá nhân
    NuScenes = None
    view_points = None
    BoxVisibility = None
    try:
        import importlib
        _nusc_mod = importlib.import_module("nuscenes.nuscenes")
        _geom_mod = importlib.import_module("nuscenes.utils.geometry_utils")
        NuScenes = getattr(_nusc_mod, "NuScenes", None)
        view_points = getattr(_geom_mod, "view_points", None)
        BoxVisibility = getattr(_geom_mod, "BoxVisibility", None)
    except Exception:
        pass

    if NuScenes is None or view_points is None or BoxVisibility is None:
        print("[THONG BAO] Thu vien 'nuscenes-devkit' chi can thiet khi tien xu ly du lieu tho tren Google Colab.")
        print("           Neu can chay buoc trich xuat du lieu nay, vui long chay: pip install nuscenes-devkit")
        return

    if not os.path.exists(dataroot):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục dữ liệu nuScenes tại: {dataroot}")
        return

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    print(f"Đang khởi tạo nuScenes ({version}) từ: {dataroot}...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    print(f"Bắt đầu trích xuất CAM_FRONT từ {len(nusc.sample)} mẫu (samples)...")
    success_count = 0
    box_count = 0

    for sample in tqdm(nusc.sample, desc="Converting nuScenes"):
        cam_token = sample['data'].get('CAM_FRONT')
        if not cam_token:
            continue

        cam_data = nusc.get('sample_data', cam_token)
        raw_img_path = os.path.join(nusc.dataroot, cam_data['filename'])

        if not os.path.exists(raw_img_path):
            continue

        # Đọc ảnh để lấy kích thước W, H
        try:
            im = Image.open(raw_img_path)
            im_w, im_h = im.size
        except Exception:
            continue

        # Tên file mới kèm tiền tố để không trùng với ảnh BDD100K
        base_name = f"{prefix}{os.path.basename(cam_data['filename'])}"
        target_img_path = os.path.join(output_img_dir, base_name)
        target_lbl_path = os.path.join(output_lbl_dir, f"{os.path.splitext(base_name)[0]}.txt")

        # Lưu ảnh sang thư mục train của YOLO
        im.save(target_img_path)

        # Trích xuất bounding boxes 3D và chiếu về 2D camera coordinates
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)

        yolo_lines = []
        for box in boxes:
            class_id = map_nuscenes_detection(box.name)
            if class_id is None or class_id not in [0, 1]:
                # Chỉ lấy 0: Vehicle và 1: Pedestrian
                continue

            # Chiếu hộp 3D sang tọa độ điểm 2D trên ảnh
            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]

            x_min = max(0.0, float(np.min(corners_2d[0, :])))
            y_min = max(0.0, float(np.min(corners_2d[1, :])))
            x_max = min(float(im_w), float(np.max(corners_2d[0, :])))
            y_max = min(float(im_h), float(np.max(corners_2d[1, :])))

            w = x_max - x_min
            h = y_max - y_min

            # Bỏ qua hộp quá nhỏ hoặc rỗng
            if w <= 15 or h <= 15:
                continue

            # Chuẩn hóa về tọa độ trung tâm và tỉ lệ [0, 1] theo chuẩn YOLOv12
            x_center = (x_min + w / 2.0) / im_w
            y_center = (y_min + h / 2.0) / im_h
            w_norm = w / im_w
            h_norm = h / im_h

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            box_count += 1

        # Ghi file nhãn .txt
        with open(target_lbl_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        success_count += 1

    print(f"\n[HOÀN TẤT] Đã trích xuất {success_count} ảnh nuScenes và tạo {box_count} bounding boxes sang {output_lbl_dir}")


# Alias tương thích tên gọi nuimages
convert_nuimages_to_yolo = convert_nuscenes_to_yolo


if __name__ == "__main__":
    # Script này dùng để tiền xử lý tập dữ liệu thô trên Google Colab
    default_root = "/content/bdd100k_raw/nuscenes-mini"
    if not os.path.exists(default_root):
        print("[THONG BAO] Script nay duoc thiet ke de tien xu ly du lieu tho nuScenes tren Google Colab.")
        print(f"           Khong tim thay duong dan: {default_root}")
        print("           Du an hien tai da co san trong so trong 'weights/', ban co the chay truc tiep Web Demo bang: .venv\\Scripts\\streamlit run src/app.py")
    else:
        convert_nuscenes_to_yolo(dataroot=default_root)
