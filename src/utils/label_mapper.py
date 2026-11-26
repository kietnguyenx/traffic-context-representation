"""
Chuẩn hóa nhãn cho các mô hình học máy (Detection YOLO & Segmentation SegFormer)
Khớp với chuẩn đề tài và configs/data.yaml
"""

# Không gian nhãn mục tiêu cho YOLO:
# 0: Vehicle (Phương tiện giao thông: ô tô, xe buýt, xe tải, xe máy,...)
# 1: Pedestrian (Người đi bộ, người đi xe đạp/rider)
# 2: Traffic Light (Đèn tín hiệu giao thông)

# =======================
# 1. QUY CHUẨN DETECTION (Phát hiện đối tượng) 
# =======================
TARGET_DETECTION_CLASSES = {
    "Vehicle": 0,
    "Pedestrian": 1,
    "Traffic Light": 2
}

# Danh sách tên nhãn theo đúng thứ tự của các ID nhãn chuẩn hóa
YOLO_DETECTION_CLASS_NAMES = ["Vehicle", "Pedestrian", "Traffic Light"]

# Ánh xạ ngược từ ID nhãn chuẩn hóa sang tên nhãn
YOLO_DETECTION_CLASS_ID_TO_NAME = {}
for class_name, class_id in TARGET_DETECTION_CLASSES.items():
    YOLO_DETECTION_CLASS_ID_TO_NAME[class_id] = class_name

# Ánh xạ nhãn từ các nhãn gốc BDD100K sang YOLO class ID
BDD_DET_MAPPING = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "train": 0,
    "trailer": 0,
    "motorcycle": 0,

    "pedestrian": 1,
    "person": 1,
    "rider": 1,
    "bicycle": 1,

    "traffic light": 2,
    "traffic_light": 2
}

# Ánh xạ nhãn từ các nhãn gốc nuScenes sang YOLO class ID
NUSCENES_DET_MAPPING = {
    "vehicle.car": 0,
    "vehicle.bus.bendy": 0,
    "vehicle.bus.rigid": 0,
    "vehicle.truck": 0,
    "vehicle.trailer": 0,
    "vehicle.construction": 0,
    "vehicle.emergency.ambulance": 0,
    "vehicle.emergency.police": 0,
    "vehicle.motorcycle": 0,

    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "human.pedestrian.construction_worker": 1,
    "human.pedestrian.police_officer": 1,
    "human.pedestrian.wheelchair": 1,
    "human.pedestrian.stroller": 1,
    "human.pedestrian.personal_mobility": 1,
    "vehicle.bicycle": 1,

    "traffic_light": 2
}

# =======================
# 2. QUY CHUẨN SEGMENTATION (Phân đoạn ngữ nghĩa)
# =======================
TARGET_SEGMENTATION_CLASSES = {
    "background": 0,            # Vùng nền, vỉa hè, bầu trời, cây cối
    "drivable_direct": 1,       # Làn đường trực tiếp đang lưu thông
    "drivable_alternative": 2,  # Làn đường chuyển hướng / chiều đối diện
    "lane_marking": 3           # Vạch kẻ sơn phân làn
}

# Danh sách tên nhãn theo đúng thứ tự của các ID nhãn chuẩn hóa
SEGMENTATION_CLASS_NAMES = ["background", "drivable_direct", "drivable_alternative", "lane_marking"]

# Ánh xạ ngược từ ID nhãn chuẩn hóa sang tên nhãn ({Tên: ID} thành {ID: Tên})
ID_TO_SEGMENTATION_CLASS = {}
for class_name, class_id in TARGET_SEGMENTATION_CLASSES.items():
    ID_TO_SEGMENTATION_CLASS[class_id] = class_name

# Bảng màu đại diện cho các nhãn phân đoạn (RGB)
SEG_COLOR_MAPPING = {
    0: [0, 0, 0],          # background: đen
    1: [46, 204, 113],     # drivable_direct: xanh ngọc lục bảo
    2: [52, 152, 219],     # drivable_alternative: xanh dương
    3: [255, 238, 0]       # lane_marking: neon cyan/vàng sáng
}

def map_bdd_detection(category_name: str):
    """
    Chuyển tên nhãn gốc từ BDD100K sang ID nhãn chuẩn hóa cho YOLO
    Trả về None nếu không tìm thấy nhãn tương ứng
    """
    if not category_name:
        return None
    return BDD_DET_MAPPING.get(category_name.strip().lower(), None)

def map_nuscenes_detection(category_name: str):
    """
    Chuyển tên nhãn gốc từ nuScenes sang ID nhãn chuẩn hóa cho YOLO
    Trả về None nếu không tìm thấy nhãn tương ứng
    """
    if not category_name:
        return None
    return NUSCENES_DET_MAPPING.get(category_name.strip().lower(), None)
