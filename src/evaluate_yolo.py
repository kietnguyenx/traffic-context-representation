"""
Evaluation & Comparison Script for Member 1 (YOLO Detection)
Dự án: Nhận diện & Biểu diễn ngữ cảnh giao thông đa nhiệm
Nhiệm vụ:
- Đo Precision, Recall, mAP@50, mAP@50:95 tổng thể và từng lớp (Vehicle, Pedestrian, Traffic Light).
- Thực hiện thí nghiệm so sánh đối chứng giữa 2 mô hình (VD: YOLOv8n vs YOLOv8s).
- Xuất file kết quả JSON/CSV và bảng Markdown cho báo cáo đồ án.
"""

import os
import sys
import argparse
import pandas as pd
from ultralytics import YOLO

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def format_markdown_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = [str(val).replace("\n", " ") for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def evaluate_single_model(weights_path, data_yaml="src/configs/data.yaml", split="val", imgsz=640, device="0"):
    print(f"\n[*] Đang đánh giá mô hình: {weights_path}...")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, plots=True, verbose=False)

    class_names = ["Vehicle", "Pedestrian", "Traffic Light"]
    data = {
        "Lớp đối tượng": ["Tất cả lớp (Overall)"] + class_names,
        "Precision": [round(float(metrics.box.mp), 4)],
        "Recall": [round(float(metrics.box.mr), 4)],
        "mAP@50": [round(float(metrics.box.map50), 4)],
        "mAP@50:95": [round(float(metrics.box.map), 4)]
    }

    for idx, cname in enumerate(class_names):
        try:
            p = float(metrics.box.p[idx])
            r = float(metrics.box.r[idx])
            m50 = float(metrics.box.ap50[idx]) if hasattr(metrics.box, 'ap50') else float(metrics.box.maps[idx][0])
            m = float(metrics.box.ap[idx])
        except Exception:
            p, r, m50, m = 0.0, 0.0, 0.0, 0.0
        data["Precision"].append(round(p, 4))
        data["Recall"].append(round(r, 4))
        data["mAP@50"].append(round(m50, 4))
        data["mAP@50:95"].append(round(m, 4))

    return pd.DataFrame(data), metrics

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình YOLO cho Thành viên 1")
    parser.add_argument("--weights", type=str, default="src/weights/yolov8_best.pt", help="File trọng số chính")
    parser.add_argument("--compare_weights", type=str, default="src/yolov8n.pt", help="File trọng số đối chứng (thí nghiệm so sánh)")
    parser.add_argument("--data", type=str, default="src/configs/data.yaml", help="File data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Split kiểm thử")
    parser.add_argument("--device", type=str, default="0", help="GPU index hoặc 'cpu'")
    parser.add_argument("--out_dir", type=str, default="reports", help="Thư mục xuất báo cáo")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("🎯 BẮT ĐẦU ĐO ĐẠC ĐỘ ĐO DETECTION (THÀNH VIÊN 1)")
    print("=" * 65)

    df_main, _ = evaluate_single_model(args.weights, args.data, args.split, device=args.device)
    print("\n📊 BẢNG CHỈ SỐ MÔ HÌNH CHÍNH:")
    print(format_markdown_table(df_main))
    df_main.to_csv(os.path.join(args.out_dir, "metrics_yolov8_main.csv"), index=False)

    if args.compare_weights and os.path.exists(args.compare_weights):
        df_comp, _ = evaluate_single_model(args.compare_weights, args.data, args.split, device=args.device)
        print("\n📊 BẢNG CHỈ SỐ MÔ HÌNH ĐỐI CHỨNG:")
        print(format_markdown_table(df_comp))
        df_comp.to_csv(os.path.join(args.out_dir, "metrics_yolo_compare.csv"), index=False)

        summary_comp = pd.DataFrame({
            "Mô hình": [os.path.basename(args.weights), os.path.basename(args.compare_weights)],
            "Precision (All)": [df_main.loc[0, "Precision"], df_comp.loc[0, "Precision"]],
            "Recall (All)": [df_main.loc[0, "Recall"], df_comp.loc[0, "Recall"]],
            "mAP@50": [df_main.loc[0, "mAP@50"], df_comp.loc[0, "mAP@50"]],
            "mAP@50:95": [df_main.loc[0, "mAP@50:95"], df_comp.loc[0, "mAP@50:95"]]
        })
        print("\n" + "=" * 65)
        print("🏆 KẾT QUẢ THÍ NGHIỆM SO SÁNH (DÀNH CHO BÁO CÁO ĐỒ ÁN):")
        print("=" * 65)
        print(format_markdown_table(summary_comp))
        summary_comp.to_csv(os.path.join(args.out_dir, "detection_comparison.csv"), index=False)

if __name__ == "__main__":
    main()
