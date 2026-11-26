"""
Module Bộ nạp và Suy luận SegFormer (SegFormer Semantic Segmentation Loader)
Hỗ trợ nạp mô hình phân đoạn ngữ nghĩa từ Thành viên 2 (SegFormer):
- Nhận diện vùng lòng đường xe chạy (Drivable Area: Direct & Alternative)
- Nhận diện vạch kẻ sơn phân làn (Lane Markings)

Tác giả: Thành viên 3 (Context Engine & Pipeline)
"""

import os
import sys
import cv2
import numpy as np

# Đảm bảo mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Thử import torch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Thử import transformers SegFormer
try:
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
        SegformerConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def safe_print(msg):
    """In log an toàn, chống crash UnicodeEncodeError trên terminal Windows"""
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode('ascii', 'replace').decode('ascii'))
        except Exception:
            pass


class SegFormerWrapper:
    """
    Wrapper chuẩn hóa cho mô hình SegFormer của Thành viên 2.
    Đầu vào: Khung hình BGR từ OpenCV (H x W x 3)
    Đầu ra: Dictionary chứa 2 mặt nạ nhị phân uint8 (0 hoặc 255):
        - masks['drivable']: Vùng đường xe chạy (Drivable Area)
        - masks['lane']: Vạch kẻ phân làn (Lane Markings)
    """

    def __init__(self, weights_path=None, device=None, num_classes=3):
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.load_error = None
        self.backbone_name = "SegFormer-B0"

        # Xác định thiết bị tính toán (CUDA nếu có, ngược lại CPU)
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device("cuda")
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            else:
                self.device = torch.device("cpu") if TORCH_AVAILABLE else "cpu"
        else:
            self.device = device

        self.use_fp16 = TORCH_AVAILABLE and str(self.device).startswith("cuda")
        if TORCH_AVAILABLE:
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self.mean_gpu = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=dtype).view(1, 3, 1, 1)
            self.std_gpu = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        if weights_path:
            self.load(weights_path)

    def reset(self):
        """Đặt lại trạng thái bộ nạp khi chuyển sang video mới"""
        pass

    def load(self, weights_path):
        """Nạp checkpoint hoặc mô hình SegFormer"""
        self.weights_path = weights_path
        if not weights_path or not os.path.exists(weights_path):
            self.is_loaded = False
            self.load_error = f"File weights khong ton tai: {weights_path}"
            return False

        if not TORCH_AVAILABLE:
            self.is_loaded = False
            self.load_error = "PyTorch chua duoc cai dat."
            return False

        try:
            safe_print(f"[SegFormerLoader] Dang nap trong so tu: {weights_path} len {self.device}...")

            # Trường hợp 1: weights_path là thư mục chứa Hugging Face SegFormer model
            if os.path.isdir(weights_path) and TRANSFORMERS_AVAILABLE:
                self.model = SegformerForSemanticSegmentation.from_pretrained(weights_path)
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                safe_print("[SegFormerLoader] Nap thanh cong SegFormer tu thu muc model.")
                return True

            # Trường hợp 2: weights_path là file checkpoint (.pth, .pt)
            checkpoint = torch.load(weights_path, map_location=self.device)

            # 2a. Checkpoint lưu trực tiếp cả đối tượng nn.Module
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                safe_print("[SegFormerLoader] Nap thanh cong PyTorch nn.Module SegFormer.")
                return True

            # 2b. Checkpoint lưu state_dict (dạng dictionary weights)
            state_dict = None
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

            if state_dict is not None and TRANSFORMERS_AVAILABLE:
                # Chuẩn hóa tên key nếu checkpoint được lưu từ DataParallel ('module.') hoặc wrapper ('model.')
                cleaned_sd = {}
                for k, v in state_dict.items():
                    new_k = k
                    if new_k.startswith("module."):
                        new_k = new_k[len("module."):]
                    if new_k.startswith("model.") and not new_k.startswith("model_"):
                        new_k = new_k[len("model."):]
                    cleaned_sd[new_k] = v
                state_dict = cleaned_sd

                # Tự động phát hiện số lớp đầu ra từ decode_head classifier weight
                if "decode_head.classifier.weight" in state_dict:
                    self.num_classes = int(state_dict["decode_head.classifier.weight"].shape[0])
                    safe_print(f"[SegFormerLoader] Phat hien so lop dau ra: {self.num_classes}")

                id2label = {i: f"class_{i}" for i in range(self.num_classes)}
                if self.num_classes == 3:
                    id2label = {0: "background", 1: "drivable_area", 2: "lane_marking"}
                elif self.num_classes == 4:
                    id2label = {0: "background", 1: "drivable_direct", 2: "drivable_alternative", 3: "lane_marking"}
                label2id = {v: k for k, v in id2label.items()}

                # Tự động nhận diện cấu hình kiến trúc Backbone (MiT-B0 -> B5) trực tiếp từ tensor weights
                # 1. Trích xuất hidden_sizes và decoder_hidden_size từ decode_head.linear_projections hoặc classifier
                dec_proj = [f"decode_head.linear_projections.{i}.proj.weight" for i in range(4)]
                if all(p in state_dict for p in dec_proj):
                    hidden_sizes = [int(state_dict[p].shape[1]) for p in dec_proj]
                    decoder_hidden_size = int(state_dict[dec_proj[0]].shape[0])
                else:
                    # Kiểm tra lớp embedding patch_embeddings của stage 0
                    c0 = 32
                    for k, v in state_dict.items():
                        if "patch_embeddings" in k and "proj.weight" in k and ("stages.0." in k or ".0." in k):
                            c0 = int(v.shape[0])
                            break
                    if c0 == 32:
                        hidden_sizes = [32, 64, 160, 256]
                        decoder_hidden_size = 256
                    else:
                        hidden_sizes = [64, 128, 320, 512]
                        decoder_hidden_size = 768

                if "decode_head.classifier.weight" in state_dict:
                    decoder_hidden_size = int(state_dict["decode_head.classifier.weight"].shape[1])

                # 2. Trích xuất độ sâu (depths) của 4 stages từ danh sách các block
                stage_blocks = {s: set() for s in range(4)}
                for k in state_dict.keys():
                    for s in range(4):
                        patterns = [f"stages.{s}.blocks.", f"stages.{s}.layers.", f"encoder.block.{s}."]
                        for pat in patterns:
                            if pat in k:
                                sub = k.split(pat)[1]
                                idx = sub.split(".")[0]
                                if idx.isdigit():
                                    stage_blocks[s].add(int(idx))
                
                # Nếu quét được block thì lấy số block thực tế, ngược lại fallback theo hidden_sizes
                if all(len(stage_blocks[s]) > 0 for s in range(4)):
                    depths = [max(stage_blocks[s]) + 1 for s in range(4)]
                else:
                    depths = [2, 2, 2, 2] if hidden_sizes[0] == 32 else [3, 4, 6, 3]

                # 3. Đặt tên hiển thị tương ứng
                if hidden_sizes == [32, 64, 160, 256] and depths == [2, 2, 2, 2]:
                    variant_name = "SegFormer-B0 (MiT-B0)"
                elif hidden_sizes == [64, 128, 320, 512]:
                    if depths == [2, 2, 2, 2]:
                        variant_name = "SegFormer-B1 (MiT-B1)"
                    elif depths == [3, 4, 6, 3]:
                        variant_name = "SegFormer-B2 (MiT-B2)"
                    elif depths == [3, 4, 18, 3]:
                        variant_name = "SegFormer-B3 (MiT-B3)"
                    elif depths == [3, 8, 27, 3]:
                        variant_name = "SegFormer-B4 (MiT-B4)"
                    elif depths == [3, 6, 40, 3]:
                        variant_name = "SegFormer-B5 (MiT-B5)"
                    else:
                        variant_name = f"SegFormer-Custom (B2-variant, depths={depths})"
                else:
                    variant_name = f"SegFormer ({hidden_sizes[0]}ch, depths={depths})"

                self.backbone_name = variant_name
                config = SegformerConfig(
                    hidden_sizes=hidden_sizes,
                    depths=depths,
                    decoder_hidden_size=decoder_hidden_size,
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    mlp_ratios=[4, 4, 4, 4],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2]
                )
                config.num_labels = self.num_classes
                config.id2label = id2label
                config.label2id = label2id
                self.model = SegformerForSemanticSegmentation(config)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                if self.use_fp16:
                    self.model.half()
                self.model.eval()
                self.is_loaded = True
                safe_print(f"[SegFormerLoader] Nap thanh cong {variant_name} state_dict ({self.device})!")
                return True

            self.load_error = "Dinh dang checkpoint chua phu hop."
            self.is_loaded = False
            return False

        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            safe_print(f"[SegFormerLoader] Loi nap SegFormer: {e}")
            return False

    def predict(self, frame_bgr):
        """
        Thực hiện suy luận phân đoạn ảnh thực tế.
        
        Args:
            frame_bgr (np.ndarray): Khung hình gốc từ OpenCV (BGR)
            
        Returns:
            dict: {"drivable": drivable_mask, "lane": lane_mask, "raw_pred": pred_map}
        """
        h_orig, w_orig = frame_bgr.shape[:2]

        if not self.is_loaded or self.model is None:
            empty = np.zeros((h_orig, w_orig), dtype=np.uint8)
            return {"drivable": empty, "lane": empty, "lane_core": empty, "raw_pred": None}

        try:
            # Tiền xử lý ảnh tỉ lệ chuẩn 16:9 (512x288 = 147,456 px, bằng đúng số điểm ảnh 384x384 nhưng nét ngang hơn 33% và không méo hình)
            input_h, input_w = 288, 512
            resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Sử dụng torch.inference_mode() và chuẩn hóa trực tiếp trên GPU bằng FP16
            inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            with inference_ctx():
                dtype = torch.float16 if self.use_fp16 else torch.float32
                tensor = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).to(dtype)
                tensor = (tensor / 255.0 - self.mean_gpu) / self.std_gpu

                outputs = self.model(tensor)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Nội suy Bilinear mượt trên GPU về đúng lưới 16:9 (512x288) để tái sử dụng ảnh đã resize
                h_mid, w_mid = input_h, input_w
                upsampled = F.interpolate(
                    logits,
                    size=(h_mid, w_mid),
                    mode="bilinear",
                    align_corners=False
                )
                pred_mid = torch.argmax(upsampled, dim=1).squeeze().to(torch.uint8).cpu().numpy()

            if self.num_classes == 3:
                # Class 1: Drivable Direct, Class 2: Drivable Alternative / Lane Corridor
                c1 = (pred_mid == 1)
                c2 = (pred_mid == 2)

                # 1. Gộp toàn bộ mặt đường xe chạy (Drivable Area) và đóng lỗ hổng
                raw_road = ((c1 | c2).astype(np.uint8)) * 255
                y_cut = int(h_mid * 0.28)
                raw_road[:y_cut, :] = 0
                road_mid = cv2.morphologyEx(raw_road, cv2.MORPH_CLOSE, self._close_k)

                # Co biên ngang mặt đường vào trong 7px để loại bỏ hoàn toàn dải phân cách bê tông / bó vỉa hai bên mép ngoài
                k_in = np.ones((3, 13), dtype=np.uint8)
                valid_road = cv2.erode(road_mid, k_in, iterations=1)

                # Vùng chú ý vạch kẻ AI (Class 2) từ SegFormer (dùng để phân biệt vạch phân làn với vạch đi bộ zebra crossing)
                c2_u8 = (c2.astype(np.uint8)) * 255
                ai_lane_zone = cv2.dilate(c2_u8, np.ones((9, 19), dtype=np.uint8), iterations=1)

                # 2. Trích xuất vạch kẻ sơn thực tế siêu tốc trên toàn bộ mặt đường hợp lệ (y >= 28%)
                roi_bgr = resized[y_cut:, :]
                roi_valid = valid_road[y_cut:, :]
                roi_ai_lane = ai_lane_zone[y_cut:, :]

                gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.blur(gray_roi, (23, 11))
                diff_white = cv2.subtract(gray_roi, bg_gray)

                # Tương phản kênh Vàng - Xanh siêu tốc ( (R+G)/2 - B )
                b_ch = roi_bgr[:, :, 0].astype(np.int16)
                g_ch = roi_bgr[:, :, 1].astype(np.int16)
                r_ch = roi_bgr[:, :, 2].astype(np.int16)
                yel_idx = np.clip(((r_ch + g_ch) >> 1) - b_ch, 0, 255).astype(np.uint8)
                bg_yel = cv2.blur(yel_idx, (23, 11))
                diff_yellow = cv2.subtract(yel_idx, bg_yel)

                mask_white = (diff_white >= 9) & (gray_roi >= 135)
                mask_yellow = (diff_yellow >= 6) & (gray_roi >= 95)

                paint_roi = ((mask_white | mask_yellow).astype(np.uint8) * 255)
                paint_roi = cv2.bitwise_and(paint_roi, roi_valid)

                # 3. Lọc hình học tuyến tính theo phối cảnh (Perspective Geometric Line Filter)
                contours, _ = cv2.findContours(paint_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                lane_roi = np.zeros_like(paint_roi)
                roi_h = float(max(1, roi_bgr.shape[0]))
                roi_w = float(max(1, roi_bgr.shape[1]))

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 3.5 or area > 1400.0:
                        continue
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (rw, rh), _ = rect
                    length = max(rw, rh)
                    if cy > roi_h * 0.42 and length < 6.5:
                        continue
                    box_thick = max(1.0, min(rw, rh))
                    true_thick = max(1.0, area / max(1.0, length))
                    thickness = min(box_thick, true_thick * 1.35)
                    aspect = length / thickness

                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Loại bỏ triệt để mảng lóa ngang ở ngã tư, vạch dừng ngang, vạch zebra và phản chiếu nắp capo ở mọi độ cao
                    if (bw > bh * 1.85 and bh < 24) or (bw > bh * 2.15) or (bw > roi_w * 0.42 and bh < roi_h * 0.28):
                        continue

                    # Nếu ở nửa dưới màn hình (vùng dễ có vạch đi bộ zebra crossing dọc), kiểm tra sự đồng thuận của AI Class 2
                    if cy > roi_h * 0.52 and length < 95.0:
                        cx_i = min(roi_ai_lane.shape[1] - 1, max(0, int(cx)))
                        cy_i = min(roi_ai_lane.shape[0] - 1, max(0, int(cy)))
                        if roi_ai_lane[cy_i, cx_i] == 0 and true_thick > 4.2:
                            continue

                    # Độ dày thực tế thích ứng theo phối cảnh: ở xa <= 5.0px, ở gần <= 12.5px
                    max_thick = 4.5 + 8.0 * (cy / roi_h)
                    if thickness <= max_thick and box_thick <= max_thick * 2.2 and (aspect >= 2.1 or (cy < roi_h * 0.38 and thickness <= 4.2 and length >= 3.0)):
                        cv2.drawContours(lane_roi, [cnt], -1, 255, -1)

                lane_mid = np.zeros_like(road_mid)
                lane_mid[y_cut:, :] = lane_roi

                # Phóng mượt lên kích thước khung hình đích của video
                drivable_up = cv2.resize(road_mid, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                lane_up = cv2.resize(lane_mid, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                _, drivable_mask = cv2.threshold(drivable_up, 127, 255, cv2.THRESH_BINARY)
                _, lane_mask = cv2.threshold(lane_up, 85, 255, cv2.THRESH_BINARY)
            else:
                road_mid = (((pred_mid == 1) | (pred_mid == 2)).astype(np.uint8)) * 255
                lane_mid = ((pred_mid == 3).astype(np.uint8)) * 255
                drivable_mask = cv2.resize(road_mid, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                lane_mask = cv2.resize(lane_mid, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            return {
                "drivable": drivable_mask,
                "lane": lane_mask,
                "raw_pred": pred_mid
            }

        except Exception as e:
            safe_print(f"[SegFormerLoader] Loi inference SegFormer: {e}")
            empty = np.zeros((h_orig, w_orig), dtype=np.uint8)
            return {"drivable": empty, "lane": empty, "lane_core": empty, "raw_pred": None}


if __name__ == "__main__":
    print("=== KIỂM THỬ TỰ ĐỘNG SEGFORMER LOADER ===")
    test_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "segformer_best.pth")
    if not os.path.exists(test_weights):
        test_weights = "weights/segformer_best.pth"
    print(f"Đang kiểm tra file trọng số: {test_weights}")
    wrapper = SegFormerWrapper(weights_path=test_weights)
    print(f"Trạng thái nạp: {wrapper.is_loaded}")
    print(f"Kiến trúc: {wrapper.backbone_name}")
    print(f"Số lượng lớp: {wrapper.num_classes}")
    if wrapper.is_loaded:
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = wrapper.predict(dummy_frame)
        print("Suy luận mẫu thành công! Kích thước mặt nạ:", res["drivable"].shape)
    else:
        print(f"Lỗi nạp: {wrapper.load_error}")
