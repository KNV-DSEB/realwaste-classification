# Giải thích code `RealWaste_Classification.ipynb` (tài liệu học, không nộp)

Đọc song song với notebook, theo thứ tự cell. Mỗi phần: cell làm gì, rồi từng dòng hoặc từng nhóm dòng. Cuối file là các câu giảng viên hay hỏi.

## Cell 2: import và hằng số

- `ROOT`, `DATA_DIR`, `SPLIT_FILE`, `CKPT_DIR`, `RESULT_DIR`: đường dẫn trong thư mục Drive chung. `Path` cho phép ghép đường dẫn bằng dấu `/`.
- `SPLIT_SHA256`: "vân tay" của file split. Nếu ai sửa file split, hash đổi và notebook dừng ngay.
- `SEEDS = [42, 43, 44]`: mỗi model E1–E3 train 3 lần với 3 seed để đo độ dao động. E4 chỉ seed 42.
- `BATCH_SIZE = 32`: mỗi bước train dùng 32 ảnh.
- `PATIENCE = 5`: 5 epoch liền không cải thiện macro-F1 trên validation thì dừng (early stopping).
- `EPOCHS`: số epoch tối đa. E3 có hai giai đoạn: 10 epoch train head, 15 epoch fine-tune.
- `NUM_WORKERS = 2`: số tiến trình đọc ảnh song song (Colab có 2 CPU).
- `device`: dùng GPU nếu có. `AMP = device.type == "cuda"`: chỉ bật mixed precision khi có GPU.

## Cell 4: đếm ảnh

- `d.glob("*.jpg")` liệt kê file `.jpg` trong mỗi thư mục lớp; `len(list(...))` là số ảnh.
- `pd.Series(counts).sort_values().plot.barh(...)` vẽ biểu đồ ngang số ảnh mỗi lớp.

## Cell 6: chia dữ liệu theo khối (grouped split)

Vấn đề: cùng một vật được chụp 2–3 lần với số file liền nhau. Chia ngẫu nhiên từng ảnh sẽ đưa ảnh này vào train, ảnh kia vào test (rò rỉ dữ liệu).

- `df.sort_values("filepath")`: sắp xếp để kết quả không phụ thuộc thứ tự đọc file.
- `str.extract(r"_(\d+)\.[A-Za-z]+$")`: lấy số cuối tên file, ví dụ `Plastic_646.jpg` → 646.
- `number // block_size`: chia nguyên cho 10 → số khối. File 640–649 cùng khối 64.
- `rng = np.random.default_rng(seed)`: bộ sinh số ngẫu nhiên có seed 42, nên chạy lại luôn ra cùng kết quả.
- Vòng `for cls in sorted(...)`: xử lý từng lớp riêng (phân tầng theo lớp, stratified). `sorted` để thứ tự cố định.
- `rng.permutation(np.sort(...unique()))`: lấy danh sách khối của lớp, sắp xếp rồi xáo trộn.
- `n_test, n_val = round(0.15 * số khối)`: 15 % số khối cho test, 15 % cho validation.
- Trong vòng `enumerate(blocks)`: các khối đầu → test, tiếp theo → val, còn lại → train. Lưu vào dict `split_of[(lớp, khối)]`.
- Dòng cuối của hàm: gán split cho từng ảnh theo cặp (lớp, khối) của nó.
- `assert hashlib.sha256(...) == SPLIT_SHA256`: kiểm tra file split không bị sửa.
- `again = grouped_split(...)` rồi `.eq(...).all()`: tạo lại split bằng code và so với file đã lưu. In `True` nghĩa là split tái tạo được đúng.

## Cell 8: dHash (kiểm tra ảnh gần trùng)

- `convert("L")`: chuyển ảnh xám. `resize((17, 16))`: thu nhỏ còn 17×16 điểm.
- `pixels[:, 1:] > pixels[:, :-1]`: so mỗi điểm với điểm bên phải → 16×16 = 256 giá trị đúng/sai (256 bit).
- `(dhash(a) != dhash(b)).sum()`: số bit khác nhau (khoảng cách Hamming). Càng nhỏ càng giống nhau.
- Tại sao dHash không bắt được hết ảnh chụp lại: vật bị xoay, lật hay dịch chỗ thì ảnh khác nhau thật, số bit khác nhau lớn. Vì vậy nhóm phải xem tay các bộ ba ảnh liền số và đổi sang chia theo khối.

## Cell 10: pipeline dữ liệu

- `mapping`: file `class_mapping.csv` gán mỗi lớp một số 0–8. `split_df["label"]` là nhãn số của từng ảnh.
- `image_path`: đường dẫn thật tới ảnh = `DATA_DIR / tên lớp / tên file`.
- `MEAN, STD`: trung bình và độ lệch chuẩn của ImageNet theo từng kênh R, G, B.
- `train_tf` (chỉ cho tập train):
  - `RandomResizedCrop(224, scale=(0.8, 1.0))`: cắt ngẫu nhiên một vùng chiếm 80–100 % diện tích ảnh rồi resize về 224×224.
  - `RandomHorizontalFlip(0.5)`: lật ngang với xác suất 50 %.
  - `RandomRotation(10)`: xoay ngẫu nhiên trong khoảng ±10°.
  - `ColorJitter(0.2, 0.2, 0.2, hue=0)`: đổi độ sáng, tương phản, bão hòa tối đa 20 %; giữ nguyên sắc màu.
  - `ToTensor()`: ảnh PIL → tensor 3×224×224, giá trị chia 255 về [0, 1].
  - `Normalize(MEAN, STD)`: (x − mean) / std cho từng kênh.
- `eval_tf` (validation, test): chỉ resize, ToTensor, Normalize. Không ngẫu nhiên, để điểm số không phụ thuộc may rủi.
- `WasteDataset`: lớp Dataset của PyTorch.
  - `__len__` trả số ảnh.
  - `__getitem__(i)` mở ảnh thứ i, đổi sang RGB, áp transform, trả `(tensor, nhãn, filepath)`. `filepath` dùng để biết ảnh nào bị đoán sai.
- `seed_everything(seed)`: đặt seed cho `random` của Python, numpy, torch trên CPU và GPU, để khởi tạo trọng số và augmentation lặp lại được.
- `seed_worker`: mỗi worker của DataLoader có seed riêng (PyTorch đặt sẵn qua `torch.initial_seed()`). Hàm này đồng bộ seed đó sang numpy và random.
- `make_loader`:
  - `shuffle=True` chỉ cho train (xáo thứ tự mỗi epoch).
  - `generator=torch.Generator().manual_seed(seed)`: thứ tự xáo lặp lại được.
  - `pin_memory`: chép dữ liệu sang GPU nhanh hơn.
  - `persistent_workers`: giữ worker sống giữa các epoch, khỏi khởi động lại.

## Cell 12: ba model

### SimpleCNN (E1)
- Vòng `for c_out in (32, 64, 128)`: mỗi khối gồm
  - `Conv2d(c_in, c_out, 3, padding=1, bias=False)`: kernel 3×3, padding 1 giữ nguyên kích thước. Không cần bias vì BatchNorm ngay sau đã có tham số dịch (beta).
  - `BatchNorm2d`: chuẩn hóa đầu ra theo mini-batch, train ổn định hơn.
  - `ReLU`: max(0, x). `inplace=True` ghi đè lên chính tensor cho đỡ tốn bộ nhớ.
  - `MaxPool2d(2)`: lấy max trong ô 2×2, chiều cao và rộng giảm một nửa.
- `AdaptiveAvgPool2d(1)`: global average pooling, mỗi feature map → 1 số (128 số).
- `classifier`: `Flatten` → `Linear(128, 128)` → ReLU → `Dropout(0.3)` → `Linear(128, 9)`.
- `forward`: features → pool → classifier. Đầu ra là logits, chưa qua softmax.
- Số tham số, ví dụ khối 1: 3×32×9 = 864 trọng số conv + 64 của BN (gamma, beta) = 928. Tổng 111,145.

### MultiScaleBlock và MultiScaleCNN (E2)
- `c1 = c5 = c_out // 4`: nhánh 1×1 và 5×5 mỗi nhánh 1/4 số kênh; nhánh 3×3 lấy phần còn lại (1/2).
- Padding 0, 1, 2 cho kernel 1, 3, 5: cả ba nhánh giữ cùng kích thước nên ghép được.
- `torch.cat([...], dim=1)`: ghép theo chiều kênh (dim 0 là batch, dim 1 là kênh).
- MultiScaleCNN: stem (conv 3×3, 32 kênh, BN, ReLU, MaxPool), rồi 4 khối đa tỉ lệ (64, 128, 256, 256), mỗi khối kèm MaxPool. Kích thước 224 → 112 → 56 → 28 → 14 → 7. Sau đó GAP và head giống E1.

### EfficientNetB0Transfer (E3)
- `efficientnet_b0(weights=IMAGENET1K_V1)`: tải mạng đã học trên ImageNet. Khi chấm test thì không cần tải (`pretrained=False`), vì trọng số lấy từ checkpoint của nhóm.
- `self.net.classifier = Dropout(0.2), Linear(1280, 9)`: thay head 1000 lớp bằng head 9 lớp.
- `set_trainable_blocks(n)`: `features` có 9 khối; khối có chỉ số `i >= 9 − n` được train, các khối khác đóng băng (`requires_grad_(False)`). n = 0: chỉ train head (giai đoạn A). n = 3: train 3 khối cuối và head (giai đoạn B).
- Hàm `train()` viết lại: PyTorch gọi `model.train()` ở đầu mỗi epoch, việc này bật BatchNorm cập nhật thống kê. Với khối đóng băng, ta ép về `eval()` để BatchNorm giữ thống kê ImageNet, và dropout ngẫu nhiên trong khối cũng tắt. Thiếu đoạn này thì "đóng băng" chỉ đúng một nửa.
- `MODELS`: bảng tra tên → lớp model. Vòng lặp in số tham số để đối chiếu với report.

## Cell 14: metric, predict, fit

### `macro_metrics`
- `precision_recall_fscore_support(..., average="macro")`: tính precision, recall, F1 từng lớp rồi lấy trung bình cộng 9 lớp. Mỗi lớp nặng như nhau.
- `labels=range(9)`: luôn tính đủ 9 lớp, kể cả lớp không xuất hiện.
- `zero_division=0`: lớp không có dự đoán nào thì precision = 0, không báo lỗi.

### `predict`
- `@torch.no_grad()`: không lưu gradient khi dự đoán (nhanh, đỡ bộ nhớ).
- `model.eval()`: tắt dropout; BatchNorm dùng thống kê đã học.
- `torch.autocast(..., enabled=AMP)`: tính bằng float16 trên GPU cho nhanh. `.float()` đưa logits về float32 trước khi tính softmax và loss cho chính xác.
- `loss += criterion(...) * len(y)`: cộng loss theo số ảnh; cuối cùng chia tổng số ảnh → loss trung bình.
- `softmax` → xác suất; `argmax` → lớp dự đoán; `max` → độ tự tin.

### `fit` (vòng train)
- `optimizer = AdamW([... if p.requires_grad], lr, weight_decay=1e-4)`: chỉ đưa tham số được train vào optimizer (quan trọng với E3).
- `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)`: macro-F1 validation không tăng 2 epoch thì nhân learning rate với 0.5.
- `GradScaler`: khi train float16, gradient nhỏ có thể bị làm tròn về 0. Scaler nhân loss lên trước khi backward rồi chia lại trước khi cập nhật trọng số.
- Khối `if last_path.exists()`: đọc lại trạng thái đã lưu (model, optimizer, scheduler, scaler, lịch sử, best, đếm stale).
  - Nếu run đã xong (stale ≥ 5 hoặc đã đủ số epoch) thì in "already finished" và dừng. Đây là lý do chạy lại notebook không train lại.
  - Nếu run bị ngắt giữa chừng, train tiếp từ epoch sau.
- Mỗi epoch:
  1. `model.train()`; với mỗi batch: `zero_grad` → forward trong autocast → `loss` → `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`. Cộng dồn loss và số dự đoán đúng.
  2. `predict` trên validation → `macro_metrics`.
  3. Ghi một dòng `history` (làm tròn 5 chữ số, kèm learning rate và thời gian epoch).
  4. Macro-F1 cao hơn best → lưu `_best.pt`, reset `stale`; không thì `stale += 1`.
  5. `scheduler.step(macro-F1)`: báo điểm cho scheduler để nó quyết định có giảm learning rate không.
  6. Lưu `_last.pt` (đủ để chạy tiếp nếu Colab ngắt) và file history CSV.
  7. `stale >= PATIENCE` → dừng sớm.

## Cell 15: chạy train

- `run_dirs`: seed 42 là lần chạy gốc, nằm thẳng trong `03_Checkpoints/E1`; seed 43, 44 nằm trong `trials/seed43`, `trials/seed44`.
- `train_run`:
  1. Tạo loader trước (E4 dùng `eval_tf` cho train, tức không augmentation).
  2. `seed_everything(seed)` rồi mới tạo model: trọng số khởi tạo phụ thuộc seed.
  3. E1, E2, E4: một lần `fit` với lr 1e-3.
  4. E3: giai đoạn A (`set_trainable_blocks(0)`, lr 1e-3, 10 epoch) → nạp trọng số tốt nhất của A → giai đoạn B (`set_trainable_blocks(3)`, lr 1e-4, 15 epoch) → giai đoạn nào có macro-F1 cao hơn thì chép thành `E3_best.pt`.
- Vòng lặp cuối: 3 seed cho E1–E3, 1 seed cho E4. Khi chạy lại, mọi run in "already finished".

## Cell 16: đường cong train
Đọc file history đã lưu, vẽ train loss, validation loss, validation macro-F1 theo epoch.

## Cell 18: chấm test
- `test_loader`: 707 ảnh test, không xáo.
- Với mỗi run: tạo model, nạp `_best.pt`, `predict` trên test, tính 4 metric.
- `stored`: đọc `metrics.json` mà notebook 09 đã lưu, đặt cạnh để chứng minh code này ra cùng con số.
- `val_macro_f1`: điểm validation lưu trong checkpoint, dùng để chọn model cho phân tích lỗi.
- `groupby("model").agg(["mean", "std"])`: trung bình và độ lệch chuẩn qua 3 seed.

## Cell 19: F1 theo lớp và confusion matrix
- `precision_recall_fscore_support(...)[2]` không có `average` → mảng F1 của từng lớp.
- `ConfusionMatrixDisplay.from_predictions`: hàng là lớp thật, cột là lớp dự đoán, ô là số ảnh.

## Cell 21: phân tích lỗi
- `idxmax()` của validation macro-F1 trung bình → model chính. Chọn theo validation, không theo test, để không "chọn sau khi xem đáp án".
- `errors`: các ảnh mà nhãn thật khác nhãn dự đoán.
- `groupby(["true", "predicted"]).size()`: đếm từng cặp nhầm lẫn.
- `-(-len(x) // 6)`: phép chia làm tròn lên (số hàng của lưới 6 cột).

## Câu giảng viên hay hỏi

1. **Vì sao dùng global average pooling thay vì flatten?** Flatten 7×7×256 ra 12,544 số, lớp dense sau đó cần hàng triệu trọng số và dễ overfit. GAP không có trọng số và chỉ còn 256 số.
2. **Vì sao conv có `bias=False`?** BatchNorm ngay sau đó trừ trung bình (làm mất bias) rồi cộng beta của nó. Bias của conv thành thừa.
3. **Vì sao không có softmax trong model?** `nn.CrossEntropyLoss` đã gồm log-softmax bên trong. Softmax chỉ dùng khi cần xác suất (trong `predict`).
4. **Vì sao chọn model theo macro-F1?** Lớp lệch nhau 2,9 lần. Accuracy có thể cao dù model bỏ sót gần hết Textile Trash; macro-F1 cho mỗi lớp trọng số như nhau.
5. **Vì sao model tự xây cũng chuẩn hóa theo ImageNet?** Để tiền xử lý giống hệt nhau giữa các model. Đây là hằng số cố định, không ước lượng từ dữ liệu của nhóm nên không rò rỉ.
6. **Vì sao giai đoạn B dùng lr 1e-4?** Các khối pretrained đã tốt sẵn. Lr lớn sẽ phá đặc trưng ImageNet; lr nhỏ chỉ chỉnh nhẹ.
7. **Vì sao phải viết lại `train()` cho EfficientNet?** Xem phần E3 ở trên: để BatchNorm của khối đóng băng không bị cập nhật.
8. **Receptive field 22 và 154 tính thế nào?** Mỗi conv 3×3 mở rộng vùng nhìn thêm 2 × (bước nhảy hiện tại); mỗi pooling nhân bước nhảy lên 2. E1: 3 → 4 → 8 → 10 → 18 → 22.
9. **Early stopping và ReduceLROnPlateau khác gì nhau?** Cả hai nhìn macro-F1 validation. Sau 2 epoch không tăng thì giảm lr một nửa; sau 5 epoch không tăng thì dừng hẳn.
10. **Vì sao train 3 seed?** Một lần train có yếu tố may rủi (khởi tạo, thứ tự batch, augmentation). 3 seed cho trung bình ± độ lệch chuẩn. Split giữ nguyên, nên SD chỉ đo độ dao động do train.
11. **Tập test dùng mấy lần?** Chỉ để chấm sau khi đã chọn xong model. Không chọn hay sửa gì dựa trên điểm test. Chạy lại notebook chỉ tính lại cùng con số từ cùng trọng số.
12. **AMP là gì, có ảnh hưởng kết quả không?** Tính bằng float16 trên GPU cho nhanh. GradScaler chống gradient bị làm tròn về 0. Kết quả gần như không đổi so với float32.
13. **Nếu Colab bị ngắt giữa chừng?** `_last.pt` lưu sau mỗi epoch; chạy lại thì `fit` đọc file đó và train tiếp, chỉ mất epoch đang chạy dở.
14. **Vì sao không tuning siêu tham số?** Không đủ thời gian. Nhóm dùng giá trị phổ biến, giống nhau cho mọi model, và nêu rõ đây là hạn chế trong report.
