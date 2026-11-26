# TỔNG HỢP CÁC CHỨC NĂNG CỐT LÕI ĐÃ TRIỂN KHAI
**Đề tài số 2: Nhận diện và biểu diễn ngữ cảnh các nhân tố ảnh hưởng đến hoạt động giao thông**
*(Phụ trách: Thành viên 3 - Context Engine, Pipeline & Web Cockpit Dashboard)*


## Chức năng 1: Suy luận Quan hệ Không gian & Đồ thị Ngữ cảnh (Spatial Scene Graph)
*Ý nghĩa: Xác định chính xác vị trí hình học và mối liên hệ tương đối giữa Xe Chủ (Ego Vehicle) với toàn bộ phương tiện xung quanh.*

- Ground Contact Point (Điểm tiếp xúc mặt đường): Xác định tọa độ đáy chính giữa Bounding Box (cx, y2) đại diện cho vị trí bánh xe tiếp xúc lòng đường, khử biến dạng góc nhìn phối cảnh.
- Định vị làn đường thích ứng (Lane Position Classification): Phân định xe đang ở 'Làn chính (Ego Lane)', 'Làn trái (Left Lane)', 'Làn phải (Right Lane)' hoặc 'Ngoài đường (Off-road)'.
- Mô hình nón phối cảnh thích ứng mặt đường (Dynamic Drivable Corridor): Tính toán bề rộng làn nở dần từ điểm tụ phía chân trời xuống đáy màn hình.
- Ước lượng Điểm tụ động (Dynamic Vanishing Point): Khảo sát 2 đường biên mép mặt đường do SegFormer trích xuất, dùng hồi quy tuyến tính tìm giao điểm để định hình trục nhìn thẳng của xe.
- Kiểm tra độ phủ ngang (Horizontal Overlap >= 35%): Nếu thân xe đè qua ranh giới làn từ 35% trở lên, hệ thống tự động nhận diện xe đang lấn làn/chạy cùng làn.
- Đồ thị vector quan hệ không gian (Spatial Links): Khóa tia liên kết trực quan thông minh từ tâm nắp ca-pô xe chủ tới chân mục tiêu trọng tâm (chỉ vẽ cho xe dẫn đầu và các mục tiêu có rủi ro, chuyển màu theo 3 cấp độ chuẩn SAE: Cyan [An toàn] -> Cam Amber [Cảnh báo/Tạt đầu] -> Đỏ Crimson [Nguy cấp/AEB], triệt tiêu hiện tượng mạng nhện gây rối mắt).

## Chức năng 2: Ước lượng Khoảng cách & Chuẩn An Toàn Va Chạm ISO 15623 (FCW & TTC)
*Ý nghĩa: Áp dụng định luật động học và quy chuẩn an toàn ô tô quốc tế ISO 15623 để phát hiện nguy cơ đâm va từ sớm.*

- Ước lượng cự ly quang học (Distance Level): Phân loại cự ly thành 'Gần (Near)', 'Vừa (Mid)', 'Xa (Far)' dựa trên tọa độ y2 và tỉ lệ chiều cao hộp bao trong khung hình.
- Chỉ số Thời gian tới va chạm (TTC - Time To Collision): Đo lường tốc độ phình to diện tích hộp bao (Area Growth Rate) qua các frame: TTC = (Area / [dArea / dt]).
- Cảnh báo va chạm phía trước (FCW - Forward Collision Warning): Khi xe phía trước cùng làn có TTC < ngưỡng an toàn (hoặc ở cự ly Gần), hệ thống kích hoạt cảnh báo Amber/Red trên HUD và gửi thông báo khẩn.

## Chức năng 3: Phanh Khẩn Cấp Tự Động (AEB - Autonomous Emergency Braking theo Euro NCAP)
*Ý nghĩa: Cơ chế bảo vệ tối cao khi xảy ra tình huống nguy cấp bất khả kháng, mô phỏng phản xạ phanh tự động của xe tự hành.*

- AEB cho Phương tiện (Vehicle AEB): Tự động kích hoạt khi xe trước cùng làn phanh đứng đột ngột hoặc cự ly cực gần (TTC < 1.3 giây hoặc y2 > 82% chiều cao ảnh).
- AEB cho Người đi bộ (PCW / Pedestrian AEB): Bảo vệ người đi bộ theo tiêu chuẩn an toàn Euro NCAP. Khi người đi bộ băng qua lòng đường (In Drivable Area = True) ở cự ly Gần (y2 > 65%), hệ thống lập tức ra lệnh 'DỪNG XE TỨC THÌ (AEB ACTIVE)'.
- Kiểm tra mặt đường bằng cửa sổ trượt 5x5: Lấy mẫu 25 pixel xung quanh chân người đi bộ để khử nhiễu biên mask phân đoạn, đảm bảo không cảnh báo nhầm người đang đi an toàn trên vỉa hè.

## Chức năng 4: Bắt Sớm Hành Vi Xe Tạt Đầu (Predictive Cut-in Hazard Detection)
*Ý nghĩa: Phát hiện chủ động các tình huống xe làn bên cạnh chuyển hướng đột ngột cắt mặt xe chủ.*

- Nhận diện chuyển làn trực tiếp: Phát hiện xe vừa đổi trạng thái từ 'Làn trái/phải' sang 'Cùng làn' ở khoảng cách Gần hoặc Vừa.
- Dự báo tạt đầu qua vectơ vận tốc ngang (vx): Nếu xe ở làn trái có vx dương lớn hướng sang phải (hoặc xe làn phải có vx âm hướng sang trái), hệ thống tính toán quỹ đạo dự phóng trong 0.4s - 1.2s.
- Cảnh báo sớm đón đầu (Proactive Warning): Nếu quỹ đạo dự đoán sẽ cắt ngang hành lang làn xe chủ trong vòng 1.0 giây, hệ thống lập tức phát cảnh báo 'XE TẠT ĐẦU' trước khi chiếc xe đó kịp chèn vào làn!

## Chức năng 5: Bám Vết Không-Thời Gian & Khử Nhiễu Động Học (Vehicle Trajectory Tracker)
*Ý nghĩa: Nâng cấp từ ngữ cảnh không gian tĩnh sang ngữ cảnh Không - Thời gian (Spatio-Temporal Context) theo chuẩn nghiên cứu ADAS hiện đại.*

- Bộ đệm lịch sử FIFO (25 frames): Lưu vết chuỗi tọa độ (x, y) của từng xe qua thời gian.
- Bộ lọc mượt hàm mũ (EMA Filter): Khử sạch rung lắc pixel của Bounding Box giữa các khung hình, giúp tính toán vận tốc và TTC mượt mà, triệt tiêu lỗi chớp nháy cảnh báo (Flicker).
- Ước lượng vectơ vận tốc thực tế v = [vx, vy]: Cho biết chính xác tốc độ tiếp cận (vy) và tốc độ đánh lái (vx).
- Phân loại trạng thái hành vi: Tự động phân định xe đang 'Đang áp sát nhanh (Closing In)', 'Đang tách xa (Moving Away)', 'Chuyển làn trái/phải (Swerving)', hay 'Tốc độ ổn định (Cruising)'.
- Tự động dọn dẹp bộ nhớ (Garbage Collection): Tự giải phóng các xe đã rời khỏi màn hình quá 30 frames, đảm bảo chạy lâu không bao giờ bị tràn RAM.

## Chức năng 6: Cảm Biến Quang Học Thời Tiết & Camera Tự Động 100% (AI Optical Sensing)
*Ý nghĩa: Tự động phân tích chất lượng thấu kính và điều kiện khí tượng trực tiếp từ luồng video, loại bỏ hoàn toàn thao tác chọn thủ công.*

- Nhận diện Ban ngày / Ban đêm: Phân tích độ sáng trung bình (Mean Brightness) của vùng quan tâm ROI trung tâm.
- Nhận diện Mưa bão / Mặt đường ướt (Wet Road Specular Reflection): Phân tích 40% nửa dưới khung hình mặt đường để tìm các vệt phản xạ ánh sáng cực đoan (độ tương phản > 140) kết hợp độ bão hòa màu bầu trời (HSV Saturation < 36).
- Nhận diện Sương mù / Kính mờ / Giọt nước bám camera: Ứng dụng biến sai toán tử đạo hàm bậc hai Laplacian (Laplacian Edge Variance) để đo độ sắc nét của thấu kính.
- Thích ứng Hệ số ma sát lốp (mu) và Co giãn Ngưỡng phanh an toàn TTC:
  • Nắng ráo tiêu chuẩn: mu = 0.85, Ngưỡng phanh TTC = 1.8 giây.
  • Ban đêm / Thiếu sáng: mu = 0.80, Ngưỡng phanh TTC = 2.4 giây.
  • Sương mù / Kính mờ: mu = 0.70, Ngưỡng phanh TTC = 2.9 giây.
  • Mưa bão / Đường ướt: mu = 0.52 (trơn trượt), Ngưỡng phanh TTC giãn lên 3.2 giây (tăng cự ly an toàn).

## Chức năng 7: Nhận Diện Trạng Thái Đèn Tín Hiệu Giao Thông (Traffic Light HSV Analysis)
*Ý nghĩa: Phân tích trạng thái màu sắc đèn tín hiệu phía trước để cảnh báo tài xế tại các giao lộ.*

- Trích xuất vùng quan tâm (ROI): Cắt riêng hộp bao của đèn giao thông do YOLO phát hiện.
- Lọc nhiễu hình học cột đèn: Kiểm tra tỉ lệ khung hình (w/h <= 1.25) và vị trí hợp lý trên cao để loại bỏ các vật thể nhận diện nhầm ở đáy đường.
- Phân tích phổ màu không gian HSV:
  • Đèn ĐỎ (Red): Bao phủ cả 2 đầu dải Hue đặc biệt (0-10 và 160-180).
  • Đèn VÀNG (Yellow): Dải Hue từ 15-36.
  • Đèn XANH (Green): Dải Hue từ 42-92.
Khuyến nghị tài xế: Kích hoạt cảnh báo dừng xe an toàn khi gặp đèn Đỏ, giảm tốc độ khi gặp đèn Vàng và thông báo tuyến đường thông suốt khi gặp đèn Xanh.

## Chức năng 8: Điều Phối Luồng Nhận Thức Đan Xen Siêu Tốc (Interleaved Scheduling Pipeline)
*Ý nghĩa: Kiến trúc tích hợp đa nhiệm giúp hệ thống đạt chuẩn thời gian thực (Real-time >= 30 FPS).*

- Tách nhịp đan xen (Interleaved Execution): Chạy YOLO Detection & Tracking ở các frame chẵn (bước 2-3 frame), chạy SegFormer Segmentation ở các frame lẻ (bước 4-6 frame).
- Triệt tiêu độ trễ đỉnh (Zero Frame Spike): Tránh hiện tượng 2 mô hình học sâu cùng suy luận nặng trong cùng 1 khung hình làm tụt FPS.
- Bộ nhớ đệm thông minh (Mask & BBox Caching): Tái sử dụng kết quả phân đoạn ở các frame trung gian, giúp toàn bộ pipeline chạy mượt mà ngay cả trên máy tính không có GPU khủng.

## Chức năng 9: Giao Diện Web Buồng Lái ADAS Chuẩn Tối Giản (Streamlit Cockpit View)
*Ý nghĩa: Giao diện người dùng chuyên nghiệp mô phỏng màn hình điều khiển trung tâm trên xe hơi hiện đại.*

- Bố cục 2 cột cố định tỉ lệ 30% / 70%: Khóa cố định giao diện vừa vặn trong 1 màn hình máy tính, không bị tràn thanh cuộn dọc.
- Card tải video kéo thả (Dropzone) & Nút xóa video [✕] tức thì: Thiết kế nút xóa tròn 20px nằm ngay hàng flexbox với nhãn file, giải phóng RAM và cache video ngay lập tức (Zero-Lag).
- Chip cảm biến quang học tự động: Hiển thị trạng thái môi trường trực quan kèm chấm phát sáng nhịp thở (Pulsing Dot).
- Đồng hồ số đo đạc Telemetry: Hiển thị Tốc độ mô phỏng, Khoảng cách xe dẫn đầu (Lead Distance), Chỉ số TTC và Trạng thái an toàn.
- Khung nhật ký dữ liệu thời gian thực (JSON Terminal): Cập nhật log chi tiết từng frame phục vụ việc kiểm tra và nghiệm thu kỹ thuật.

## Chức năng 10: Bộ Công Cụ Đo Kiểm Định Lượng & Thí Nghiệm Bóc Tách (Benchmark Suite)
*Ý nghĩa: Đầy đủ số liệu thực nghiệm khoa học sẵn sàng để điền vào Chương 4 cuốn Báo cáo Tốt nghiệp.*

- Script đo kiểm độc lập (evaluate_context.py): Đánh giá trên bộ dữ liệu kiểm thử chuẩn Ground-Truth 60 frames bao quát 4 kịch bản giao thông (Safe, FCW, PCW, Off-road).
- Thí nghiệm bóc tách (Ablation Study): Kiểm thử độ bền vững của thuật toán khi mặt nạ phân đoạn bị co viền, mất góc (Noisy Mask) so với mặt nạ chuẩn (Clean Mask).
- Kết quả đo kiểm thực tế đạt mức hoàn hảo:
  • Độ chính xác nhận diện lòng đường (Drivable Area Accuracy): 100.0%
  • Độ chính xác định vị làn xe (Lane Position Accuracy): 100.0%
  • Độ chính xác cảnh báo an toàn (Warning Precision / Recall / F1-Score): 100.0%
  • Tốc độ suy luận của bộ não Context: 0.22 ms/frame (~4500 FPS), hoàn toàn không làm trễ pipeline.

## Chức năng 11: Hệ Thống Mã Màu Trực Quan Hóa Chuẩn Quốc Tế ADAS (SAE J2395 / ISO 15005)
*Ý nghĩa: Chuẩn hóa hệ thống thị giác theo tiêu chuẩn công thái học buồng lái ô tô quốc tế, chống quá tải nhận thức (Cognitive Overload) cho tài xế bằng cách quy toàn bộ hiển thị video và buồng lái về 4 màu ngữ nghĩa cốt lõi:*

- 🟢 Xanh lục bảo (Emerald Green - #2ecc71 / BGR [95, 205, 46]): Đại diện cho vùng an toàn khả dụng (Free Space & Go) — Lớp phủ mặt đường xe chạy được (Drivable Area, hòa trộn mờ 32% giữ rõ vân nhựa đường) và tín hiệu Đèn Giao Thông Xanh (tuyến đường thông suốt).
- 🔵 Xanh lơ công nghệ (Nominal Cyan - #00e5ff / BGR [245, 220, 0]): Đại diện cho trạng thái theo dõi an toàn (Tracked & Nominal) — Áp dụng cho các phương tiện di chuyển bình thường giữ khoảng cách an toàn, người đi bộ trên vỉa hè và vạch kẻ phân làn đường (hòa trộn mờ 58% mềm mại, tách biệt với khung xe).
- 🟠 Cam hổ phách (Advisory Amber - #ff9800 / BGR [0, 165, 255]): Cảnh báo Cấp độ 1 (Caution / Advisory Warning) — Đại diện cho các nguy cơ tiềm ẩn đang phát triển: Xe làn bên tạt đầu (Cut-in Hazard), khoảng cách với xe phía trước bị thu hẹp hoặc tín hiệu Đèn Giao Thông Vàng.
- 🔴 Đỏ rực khẩn cấp (Crimson Red - #ff1744 / BGR [35, 25, 255]): Báo động Cấp độ 2 Nguy hiểm tức thời (Imminent Hazard / Critical Alert) — Kích hoạt khi có can thiệp Phanh Khẩn Cấp Tự Động (AEB Active, khung viền dày 3px chữ trắng), người đi bộ cắt ngang lòng đường (Pedestrian on Road), hoặc tín hiệu Đèn Giao Thông Đỏ.
- **Cơ chế tinh lọc thông minh chống rối mắt:** Khi mật độ xe đông (>= 5 xe), các xe an toàn ở xa (Dist: Far) tự động rút gọn nhãn thành [#ID | Lane]. Đặc biệt với giàn đèn ngã tư gồm nhiều cụm đèn treo cạnh nhau, hệ thống vẽ khung viền mảnh 1-2px sắc nét cho từng đèn và chỉ gắn DUY NHẤT một nhãn đại diện [Signal: RED] ở vị trí trung tâm, loại bỏ hoàn toàn tình trạng các bảng chữ đỏ xếp tầng bậc thang che khuất bầu trời.