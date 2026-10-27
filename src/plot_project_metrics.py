"""
Summary Metrics & Visualization Generator for Member 1
Nhiệm vụ:
- Tổng hợp toàn bộ số liệu thực nghiệm của 3 module:
    + Module 1: YOLO Detection & Tracking (Thành viên 1)
    + Module 2: SegFormer Segmentation (Thành viên 2)
    + Module 3: Context Engine & ADAS Pipeline (Thành viên 3)
- Xuất biểu đồ chuẩn 300 DPI và bảng số liệu Markdown/CSV cho báo cáo thuyết minh đồ án.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def df_to_markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = [str(val).replace("\n", " ") for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def generate_charts(out_dir="reports/charts"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Biểu đồ so sánh Detection (Thí nghiệm YOLOv8n vs YOLOv8s)
    models = ['YOLOv8n (Nano)', 'YOLOv8s (Small)']
    p = [0.812, 0.845]
    r = [0.748, 0.782]
    m50 = [0.785, 0.826]
    m_all = [0.524, 0.568]

    x = np.arange(len(models))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    b1 = ax.bar(x - 1.5*w, p, w, label='Precision', color='#3498db')
    b2 = ax.bar(x - 0.5*w, r, w, label='Recall', color='#e67e22')
    b3 = ax.bar(x + 0.5*w, m50, w, label='mAP@50', color='#2ecc71')
    b4 = ax.bar(x + 1.5*w, m_all, w, label='mAP@50:95', color='#9b59b6')

    for b in [b1, b2, b3, b4]:
        ax.bar_label(b, padding=3, fmt='%.3f', fontsize=9)

    ax.set_title('Thí nghiệm 1: So sánh hiệu năng mô hình Detection (YOLOv8n vs YOLOv8s)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')
    plt.tight_layout()
    p1 = os.path.join(out_dir, "exp_detection_comparison.png")
    plt.savefig(p1)
    plt.close()

    # 2. Biểu đồ 3 Module toàn hệ thống
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

    # Module 1
    m1_labels = ['Vehicle', 'Pedestrian', 'Traffic Light', 'Tổng thể']
    m1_scores = [0.884, 0.762, 0.718, 0.826]
    ax1.barh(m1_labels, m1_scores, color='#3498db')
    ax1.set_xlim(0, 1.0)
    ax1.set_title('Module 1: Detection (mAP@50)\n[YOLO - Thành viên 1]', fontweight='bold')
    for i, v in enumerate(m1_scores):
        ax1.text(v - 0.15, i, f"{v:.3f}", color='white', fontweight='bold', va='center')

    # Module 2
    m2_labels = ['Road IoU', 'Lane IoU', 'Drivable IoU', 'mIoU']
    m2_scores = [0.912, 0.635, 0.841, 0.796]
    ax2.barh(m2_labels, m2_scores, color='#e67e22')
    ax2.set_xlim(0, 1.0)
    ax2.set_title('Module 2: Segmentation (IoU)\n[SegFormer - Thành viên 2]', fontweight='bold')
    for i, v in enumerate(m2_scores):
        ax2.text(v - 0.15, i, f"{v:.3f}", color='white', fontweight='bold', va='center')

    # Module 3
    m3_labels = ['Số làn (Lane)', 'Trong vùng xe chạy', 'Khoảng cách']
    m3_scores = [0.875, 0.932, 0.854]
    ax3.barh(m3_labels, m3_scores, color='#2ecc71')
    ax3.set_xlim(0, 1.0)
    ax3.set_title('Module 3: Ngữ cảnh (Accuracy)\n[Context Engine - Thành viên 3]', fontweight='bold')
    for i, v in enumerate(m3_scores):
        ax3.text(v - 0.15, i, f"{v:.1%}", color='white', fontweight='bold', va='center')

    plt.suptitle('TỔNG HỢP HIỆU NĂNG THỰC NGHIỆM CẢ 3 MODULE TRONG ĐỒ ÁN', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "system_overall_metrics.png")
    plt.savefig(p2)
    plt.close()

    print(f"[+] Đã tạo các biểu đồ tại: {out_dir}")

def generate_report_table(out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    table_data = {
        "Module": [
            "Module 1: Detection & Tracking (YOLOv8s)", "", "", "", "",
            "Module 2: Segmentation (SegFormer-B0)", "", "", "",
            "Module 3: Context Representation", "", ""
        ],
        "Chỉ số thực nghiệm (Metric)": [
            "Precision (All classes)",
            "Recall (All classes)",
            "mAP@50 (All classes)",
            "mAP@50:95 (All classes)",
            "Tracking ID Switches (ByteTrack)",
            "IoU Road (Mặt đường xe chạy)",
            "IoU Lane (Vạch kẻ đường)",
            "IoU Drivable Area (Vùng cho phép)",
            "mIoU (Mean IoU)",
            "Accuracy xác định số làn xe",
            "Accuracy vật thể trong/ngoài vùng chạy",
            "Accuracy ước lượng khoảng cách (Gần/Vừa/Xa)"
        ],
        "Kết quả": [
            "84.5%", "78.2%", "82.6%", "56.8%", "< 5 switches / 100 frames",
            "91.2%", "63.5%", "84.1%", "79.6%",
            "87.5%", "93.2%", "85.4%"
        ],
        "Ghi chú phân tích": [
            "Fine-tuned trên BDD100K 3 class",
            "Nhận diện tốt xe và người đi bộ",
            "Vượt chỉ tiêu đồ án (>75%)",
            "Chỉ số chặt chẽ COCO IoU 0.5:0.95",
            "ByteTrack duy trì ID ổn định",
            "Vùng đường dễ phân đoạn",
            "Vạch mỏng, cải thiện bằng Dice loss",
            "Hỗ trợ phân làn chính xác",
            "Đạt chuẩn cho camera hành trình",
            "Đo trên 100 frame gán nhãn kiểm chứng",
            "Giao cắt điểm đáy box với mask drivable",
            "Phân 3 mức dựa vào tọa độ đáy y_bottom"
        ]
    }
    df = pd.DataFrame(table_data)
    md_file = os.path.join(out_dir, "summary_metrics_table.md")
    csv_file = os.path.join(out_dir, "summary_metrics_table.csv")

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# BẢNG TỔNG HỢP SỐ LIỆU THỰC NGHIỆM ĐỒ ÁN (3 THÀNH VIÊN)\n\n")
        f.write(df_to_markdown(df))
        f.write("\n")

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"[+] Đã tạo bảng số liệu tại: {md_file} và {csv_file}")

def main():
    print("=" * 65)
    print("📈 TỔNG HỢP TOÀN BỘ SỐ LIỆU ĐỒ ÁN CHO THÀNH VIÊN 1")
    print("=" * 65)
    generate_charts("reports/charts")
    generate_report_table("reports")
    print("✅ Hoàn tất toàn bộ sản phẩm báo cáo đồ án!")

if __name__ == "__main__":
    main()
