# Bàn giao dự án RealWaste — 05/10/2026

Tài liệu này dành cho cả nhóm, để bất kỳ ai cũng tiếp tục được dự án mà không cần hỏi lại. Đọc mục 1–3 trước; các mục sau dùng để tra cứu khi làm.

Mốc thời gian dưới đây giả định Ngày 1 = 02/10: **code freeze 08/10 · result freeze 09/10 · báo cáo xong 10/10 · nộp 11/10**.

---

## 1. Tóm tắt trong một phút

- **Đề tài:** phân loại 9 loại rác thật (RealWaste, 4752 ảnh) bằng ba mô hình CNN, so sánh công bằng trên cùng một quy trình.
  - **E1 SimpleCNN:** CNN đơn giản do nhóm tự thiết kế, train từ đầu.
  - **E2 MultiScaleCNN:** CNN phức tạp do nhóm tự thiết kế (khối đa tỉ lệ 1×1/3×3/5×5), train từ đầu.
  - **E3 EfficientNet-B0:** transfer learning từ ImageNet; giai đoạn A chỉ train head, giai đoạn B fine-tune 3 khối cuối.
  - **E4 (tùy chọn, đã chạy):** E2 nhưng tắt augmentation, để đo đóng góp của augmentation.
- **Đã xong:**
  - kiểm tra dữ liệu (audit);
  - phát hiện và xử lý rò rỉ dữ liệu: cùng một vật được chụp nhiều lần, nên split được chia theo khối số file;
  - pipeline dùng chung;
  - train E1–E4 và đánh giá trên tập validation.
- **Đang làm:** chạy lại E1, E2, E3 với seed 43 và 44, giữ nguyên cấu hình (D030). Không dò siêu tham số nữa vì không kịp thời gian.
- **Còn lại:** chạy seed → đánh giá cuối trên tập test (một lần duy nhất) → phân tích lỗi → báo cáo → slide → nộp.

## 2. Kết quả hiện có (validation — chưa phải kết quả cuối)

| Mô hình | Accuracy | Macro-F1 | Số tham số | Ghi chú |
|---|---|---|---|---|
| E1 SimpleCNN | 0,656 | 0,650 | 111 nghìn | dừng sớm ở epoch 38, tốt nhất ở epoch 33; không overfit, mô hình quá nhỏ |
| E2 MultiScaleCNN | 0,787 | 0,787 | 1,23 triệu | 40 epoch, tốt nhất ở epoch 35; overfit nhẹ ở cuối |
| E3 EfficientNet-B0 | **0,880** | **0,879** | 4,02 triệu | giai đoạn A đạt 0,810 → giai đoạn B đạt 0,879 |
| E4 (E2 không augmentation) | 0,814 | 0,808 | 1,23 triệu | hơn E2 0,021 nhưng nằm trong nhiễu, nên **không kết luận** (D028) |

- Đây là số trên **validation**, dùng để chọn mô hình. **Tập test chưa được dùng.** Số đưa vào báo cáo là số test từ notebook 09.
- Validation dao động khoảng ±0,04 macro-F1 giữa các epoch, nên chênh lệch nhỏ hơn mức này không có ý nghĩa.
- Chi tiết từng lần chạy (log, recall theo lớp, nhầm lẫn) nằm trong `experiments/E*/README.md`.

## 3. Việc còn lại — ai làm gì

Người phụ trách dựa theo vai trò trong `docs/TEAM_OPERATING_SYSTEM.md`. Điền tên vào cột trống.

| # | Việc | Vai trò đề xuất | Tên | Hạn | Đầu ra |
|---|---|---|---|---|---|
| 1 | Chạy seed 43/44 cho **E1** (notebook 10, `EXP = "E1"`) | Simple CNN Lead | | 09/10 | `04_Results/experiments/E1/selection.json` |
| 2 | Chạy seed 43/44 cho **E2** (`"E2"`) | Complex CNN Lead | | 09/10 | `.../E2/selection.json` |
| 3 | Chạy seed 43/44 cho **E3** (`"E3"`) | Transfer/Evaluation Lead | | 09/10 | `.../E3/selection.json` |
| 4 | Ghi macro-F1 validation của 3 seed vào `docs/DECISION_LOG.md` (dòng mới D032) | Integration Lead | | 09/10 | các dòng DECISION_LOG |
| 5 | **Đánh giá cuối:** notebook 09 — chạy ngay được (seed 42), chạy lại sau notebook 10 để thêm seed 43/44 | Transfer/Evaluation Lead | | 09/10 | `04_Results/final/model_comparison.csv` … |
| 6 | Ghi **result freeze** vào DECISION_LOG (số test, ngày chạy) | Integration Lead | | 09/10 | dòng DECISION_LOG |
| 7 | Phân tích lỗi: 50–100 ảnh test bị đoán sai của mô hình chính | Cả nhóm | | 09/10 | `{id}_errors.csv` đã điền |
| 8 | Báo cáo (mục 6) | Cả nhóm | | 10/10 | file báo cáo |
| 9 | Slide và tập thuyết trình | Cả nhóm | | 11/10 | slide |
| 10 | Đối chiếu mọi con số trong báo cáo/slide với file kết quả; đóng gói nộp | Integration Lead | | 11/10 | bản nộp, commit cuối |

Việc 1–3 chạy **song song**, mỗi người trên tài khoản Colab của mình. Phần lý thuyết, dataset, tiền xử lý và kiến trúc trong báo cáo có thể **viết ngay**, không cần chờ kết quả.

## 4. Chạy seed 43/44 (notebook 10)

**Cần làm một lần — chia sẻ Drive:**
- Người sở hữu chia sẻ thư mục `Deep Learning - RealWaste` cho từng thành viên với quyền **Editor**.
- Mỗi thành viên mở thư mục đó trong Drive → **Organize → Add shortcut → My Drive**, và **giữ nguyên tên thư mục**. Nhờ vậy đường dẫn `/content/drive/MyDrive/Deep Learning - RealWaste` sẽ tồn tại khi chạy trên Colab.

**Mỗi người, cho mô hình của mình:**
1. Mở `notebooks/10_seed_runs.ipynb` từ GitHub trên Colab, rồi chọn **Runtime → Change runtime type → T4 GPU**.
2. Sửa dòng đầu của ô thứ 4: `EXP = "E1"` (hoặc `"E2"`, `"E3"`). **Không sửa gì khác.**
3. Chọn **Runtime → Run all**.
   - Mỗi phiên, notebook chép ảnh từ Drive về ổ cục bộ một lần (khoảng 15 phút), sau đó train nhanh.
   - Notebook train lại mô hình **với đúng cấu hình cũ**, lần lượt seed 43 rồi seed 44. Seed 42 là lần chạy đã có từ notebook 05–07.
4. Nếu Colab bị ngắt hoặc hết quota GPU, **cứ Run all lại**: seed đã xong được bỏ qua, seed dở dang chạy tiếp từ checkpoint.
5. Khi xong, ô cuối in ra bảng 3 lần chạy (base, seed43, seed44) và `selection` (macro-F1 trung bình ± SD).
   - Gửi hai thứ đó vào nhóm.
   - Lưu notebook lên GitHub bằng **File → Save a copy in GitHub**, đặt tên **`notebooks/10_seed_runs_E1.ipynb`** (hoặc E2, E3), để không ghi đè lên nhau.

**Thời gian GPU ước tính** (chưa tính khoảng 15 phút chép ảnh mỗi phiên):

| Mô hình | Số lần chạy | Ước tính |
|---|---|---|
| E1 | 2 (seed 43, 44) | ~50 phút (~40 giây/epoch, ~38 epoch mỗi lần) |
| E2 | 2 | ~1 giờ (~45 giây/epoch, tối đa 40 epoch) |
| E3 | 2 | ~35 phút (giai đoạn A + B, ~25 epoch mỗi lần) |

**Quy tắc:**
- **Mỗi mô hình chỉ một người chạy.** Hai phiên cùng chạy một mô hình sẽ ghi đè thư mục của nhau.
- Notebook chỉ dùng **validation**. Tập test không được nạp.

## 5. Đánh giá cuối trên tập test (notebook 09)

- **Khi nào chạy:** trước hay sau notebook 10 đều được (D031).
  - Chạy trước: chấm test cho lần chạy seed 42 của E1–E4 (cột SD để trống), in dòng "seeds 43/44 not run yet".
  - Sau khi cả 3 người chạy xong notebook 10, chạy 09 **lại một lần nữa**: thêm seed 43/44, ra trung bình ± SD cho E1–E3. Kết quả seed 42 được dùng lại, không chấm lại.
- **Mỗi checkpoint chỉ được chấm một lần:**
  - Chạy lại thì dùng kết quả đã lưu.
  - Nếu ai đó sửa checkpoint sau khi đã đánh giá, notebook dừng.
- **Sau khi chạy:**
  - Gửi bảng so sánh, bảng F1 theo lớp và danh sách cặp nhầm lẫn ở ô cuối.
  - Lưu notebook kèm output lên GitHub.
  - Ghi **result freeze** vào DECISION_LOG sau lần chạy có đủ 3 seed.
- **Sau result freeze, không thay đổi bất cứ thứ gì dựa trên số test:** không đổi mô hình, siêu tham số hay split (CONSTITUTION C3).
- **Phân tích lỗi:**
  - Mô hình chính là mô hình có macro-F1 trung bình cao nhất trên **validation**.
  - Mở `04_Results/experiments/{id}/{id}_errors.csv` của mô hình đó.
  - Điền `failure_mode` và `visual_note` cho 50–100 ảnh, nhóm theo cặp nhầm lẫn.

## 6. Báo cáo — viết gì và lấy số ở đâu

**Nguyên tắc:** mọi con số trong báo cáo phải lấy từ file kết quả. Không ước lượng, không làm tròn khác đi (C4).

| Mục báo cáo (theo yêu cầu giảng viên) | Nội dung chính | Nguồn |
|---|---|---|
| Lý thuyết CNN | convolution, pooling, FC, GAP, BatchNorm, dropout, loss/optimizer, augmentation, transfer learning | `docs/REPORT_MAPPING.md` |
| Bài toán | mục tiêu, đầu vào/đầu ra, 4 câu hỏi nghiên cứu | `docs/PROJECT_SPEC.md` §1–5 |
| Dataset | 4752 ảnh, 9 lớp, 524×524 RGB, số ảnh mỗi lớp, mất cân bằng 2,9×, ảnh mẫu | `docs/RESEARCH_STATE.md`, `report/dataset_audit_notes.md`, `04_Results/dataset_audit/sample_grid.png` (Drive) |
| Tiền xử lý | **làm sạch:** 0 ảnh lỗi, kiểm tra trùng lặp; **chia dữ liệu:** split theo khối và lý do (D014–D017), 3326/719/707; **normalization** ImageNet (D008); **augmentation** (D011) | DECISION_LOG D014–D017, D008, D011 |
| Mô hình | sơ đồ và số tham số E1/E2/E3; quy trình 2 giai đoạn của E3 | `experiments/E*/README.md`, D013, D020, D022 |
| Thí nghiệm | cấu hình train chung (cố định từ trước, giống nhau cho mọi mô hình); 3 seed mỗi mô hình; đường cong train/val | `04_Results/experiments/E*/E*_seed_runs.csv`, `*_curves.png` (Drive) |
| Kết quả | bảng test (mean ± SD qua 3 seed), F1 theo lớp, confusion matrix, E4 so với E2 (RQ4), phân tích lỗi | `04_Results/final/` và `04_Results/experiments/E*/` (Drive) |
| Kết luận | trả lời RQ1–RQ4 bằng số test; nêu hạn chế (bên dưới) | |
| Tài liệu tham khảo | chỉ trích nguồn đã đọc; kiểm tra lại từng trích dẫn | |

**Điểm mạnh nên làm nổi bật:**
- Phát hiện dataset có nhiều ảnh chụp lại cùng một vật (17/32 bộ ba ảnh liền số). Split ngẫu nhiên theo ảnh sẽ làm rò rỉ dữ liệu; nhóm đã đo và sửa bằng split theo khối.
- Quy trình so sánh công bằng: cùng split, tiền xử lý và cách đánh giá; mỗi mô hình 3 seed; tập test chỉ dùng một lần.
- Phép thử E4 và kết quả trung thực của nó.
- Chỉ riêng giai đoạn A của E3 (đặc trưng ImageNet đóng băng) đã vượt E2 trên validation.

**Hạn chế phải nêu:**
- Vẫn có thể còn rò rỉ ở ranh giới khối, hoặc ảnh cùng vật nằm xa nhau về số file (D017).
- Chỉ dùng một dataset, một độ phân giải (224×224).
- Không dò siêu tham số (D030): cấu hình chọn từ trước theo giá trị thông dụng, có thể chưa tối ưu, nhất là với E1.
- SD qua 3 seed chỉ đo độ dao động do ngẫu nhiên khi train, không đo dao động do cách chia dữ liệu. E4 chỉ có 1 seed.
- So sánh E2 với E1 không tách riêng được hiệu quả của tính đa tỉ lệ (D020).
- So sánh E3 với E1/E2 là transfer learning so với train từ đầu, không phải so sánh thuần kiến trúc (D022).

**Gợi ý tài liệu tham khảo** (đọc và kiểm tra lại trước khi trích):
- bài báo gốc RealWaste (Single và cộng sự, 2023);
- EfficientNet (Tan & Le, 2019);
- GoogLeNet/Inception (Szegedy và cộng sự, 2015), cho ý tưởng nhánh đa tỉ lệ;
- Batch Normalization (Ioffe & Szegedy, 2015);
- Dropout (Srivastava và cộng sự, 2014);
- AdamW (Loshchilov & Hutter, 2019).

## 7. Bản đồ repo và Drive

**Notebook** (thư mục `notebooks/`, chạy trên Colab):

| Notebook | Việc | Trạng thái |
|---|---|---|
| `00_setup` | tải dataset, chép lên Drive | xong |
| `01_dataset_audit` | kiểm tra dữ liệu | xong |
| `02_split_generator` | split cũ theo ảnh (D007) | **đã bị thay — không chạy lại** |
| `02b_grouped_split` | split theo khối (D016) | xong — **không chạy lại** |
| `03_duplicate_check` | kiểm tra ảnh trùng | xong |
| `04_pipeline_smoke_test` | kiểm tra pipeline | xong |
| `05`–`08_train_E1…E4` | train E1–E4 (lần thử `base`) | xong |
| `10_seed_runs` | chạy seed 43/44 | **cần chạy** (mục 4) |
| `09_final_evaluation` | đánh giá test một lần | **chạy được ngay; chạy lại sau notebook 10** (mục 5) |

**Code** (`src/`):
- `dataset.py`, `transforms.py`: dữ liệu, augmentation, kiểm tra hash của split.
- `models/`: E1, E2, E3 và `build_model`.
- `train.py`: vòng train, early stopping, checkpoint, chạy tiếp.
- `evaluate.py`: metric.
- `tuning.py`: chạy các lần thử/seed và ghi `selection.json`.
- `final_eval.py`: đánh giá test.

Mọi thiết lập nằm trong `configs/config.yaml`.

**Tài liệu** (`docs/`):
- `DECISION_LOG.md`: mọi quyết định D001–D031.
- `RESEARCH_STATE.md`: trạng thái đã kiểm chứng.
- `RESULT_CONTRACT.md`: tên và nội dung các file kết quả.
- `PROJECT_SPEC.md`: đặc tả.

**Drive** (`Deep Learning - RealWaste/`):
```
01_Dataset/RealWaste/             ảnh gốc (4752)
02_Splits/split_grouped_v2.csv    split đang dùng (SHA256 65dc85e8…) — KHÔNG sửa
02_Splits/split.csv               split cũ D007 — chỉ để lưu vết
02_Splits/class_mapping.csv       0 Cardboard … 8 Vegetation
03_Checkpoints/E1…E4/             checkpoint; trials/seed43, trials/seed44 cho các seed
04_Results/dataset_audit/         kết quả audit và kiểm tra trùng lặp
04_Results/experiments/E1…E4/     kết quả train, seed, test
04_Results/final/                 bảng so sánh cuối (sau notebook 09)
```

## 8. Luật không được phá

1. **Tập test chỉ được dùng ở notebook 09; mỗi checkpoint chấm một lần.** Không chỉnh gì sau khi đã thấy số test.
2. **Không sửa, ghi đè hay tạo lại file split.** Không chạy lại `02` hay `02b`. Pipeline kiểm tra SHA256 và sẽ dừng nếu split thay đổi.
3. **DECISION_LOG chỉ thêm dòng mới, không sửa dòng cũ.** Mọi thay đổi về mô hình, siêu tham số hay dữ liệu phải ghi vào đó trước.
4. **Mỗi mô hình một người khi chạy seed.**
5. **Mọi con số trong báo cáo lấy từ file kết quả.** Không ghi "khoảng", không tự tính lại khác đi.
6. **Trước khi commit, luôn `git pull`,** nhất là sau khi vừa lưu notebook từ Colab.

## 9. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Epoch đầu mất 15–17 phút | Bình thường: lần đầu đọc ảnh từ Drive rất chậm. Notebook 10 chỉ chép một lần mỗi phiên. |
| Colab bị ngắt hoặc hết quota GPU | Run all lại (có thể hôm sau, hoặc dùng tài khoản thành viên khác đã có shortcut Drive). Notebook chạy tiếp từ checkpoint; điều này đã kiểm chứng thật với E4 (ngắt ở epoch 27, chạy tiếp từ 28). |
| Notebook báo "already finished" hoặc chạy code cũ | Runtime → Restart session, rồi Run all. (Từ notebook 06 trở đi đã tự xử lý.) |
| `git push` bị từ chối (non-fast-forward) | `git pull --rebase origin main`, rồi `git push origin main`. |
| Không tìm thấy thư mục Drive | Kiểm tra shortcut trong My Drive có đúng tên `Deep Learning - RealWaste` không. |
| Notebook 09 in "seeds 43/44 not run yet" | Bình thường nếu notebook 10 chưa xong; chạy lại 09 sau khi xong. |
| Notebook 09 báo "changed after it was evaluated" | Có người đã sửa checkpoint sau khi đánh giá. Dừng lại và báo cả nhóm, không tự xử lý. |
| "SHA256 no longer matches" khi nạp split | File split trên Drive bị thay đổi. Dừng lại và báo cả nhóm. |

## 10. Lịch sử quyết định chính

Chi tiết đầy đủ ở `docs/DECISION_LOG.md`.

- **Dữ liệu và split:**
  - **D006** kết quả audit.
  - **D007** split cũ theo ảnh.
  - **D014** phát hiện chụp lại cùng vật.
  - **D015/D016** split theo khối 10 số file liên tiếp.
  - **D017** Gate 2 PASS.
- **Giao thức chung:**
  - **D008** normalization ImageNet.
  - **D009/D024** CE không trọng số.
  - **D010** chọn mô hình theo macro-F1 trên validation.
  - **D011** augmentation.
  - **D012** quy tắc đặt tên.
- **Mô hình:**
  - **D018/D019** cấu hình train E1, trần 40 epoch chung cho E1/E2.
  - **D013/D020** đặc tả E2.
  - **D022** đặc tả E3.
- **Kết quả train:**
  - **D021** E1.
  - **D023** E2.
  - **D027** E3.
  - **D025/D028** E4.
- **Đánh giá và seed:**
  - **D026** quy trình đánh giá test một lần.
  - **D029** tinh chỉnh 3 pha (đã bỏ phần dò siêu tham số).
  - **D030** không tuning; chỉ chạy thêm seed 43/44 với cấu hình cũ.
  - **D031** notebook 09 chạy được trước notebook 10.
