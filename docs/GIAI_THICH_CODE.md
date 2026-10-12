# Giải thích từng dòng code `RealWaste_Classification.ipynb`

Tài liệu học, không nộp. Đọc song song với notebook. Mỗi dòng code được chép lại, bên dưới là **làm gì** và **vì sao**. Số "Cell" là thứ tự ô trong notebook, đếm từ 0, tính cả ô markdown.

---

## Cell 1: kết nối Google Drive

```python
from google.colab import drive
```
Lấy module `drive` của Colab. Module này chỉ có trên Colab.

```python
drive.mount('/content/drive')
```
Gắn Google Drive vào máy Colab như một thư mục thường tại `/content/drive`. Colab sẽ hỏi quyền truy cập. Phải mount vì ảnh, file split và checkpoint đều nằm trên Drive chung của nhóm; máy Colab bị xoá sạch sau mỗi phiên.

---

## Cell 2: import thư viện và hằng số

```python
import json, random, shutil, time
```
Bốn module có sẵn của Python:
- `json`: đọc file `metrics.json` đã lưu.
- `random`: bộ sinh số ngẫu nhiên của Python; phải đặt seed cho nó vì một số phép augmentation dùng nó.
- `shutil`: chép file (dùng để chép checkpoint tốt nhất của E3).
- `time`: đo thời gian mỗi epoch.

```python
from pathlib import Path
```
`Path` biểu diễn đường dẫn dạng đối tượng, ghép bằng `/` (ví dụ `ROOT / "03_Checkpoints"`), chạy giống nhau trên mọi hệ điều hành.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
```
- `plt`: vẽ biểu đồ.
- `np`: tính toán mảng số.
- `pd`: bảng dữ liệu (DataFrame), dùng cho file split và bảng kết quả.
- `torch`: PyTorch, thư viện deep learning chính.

```python
from PIL import Image
```
Mở file ảnh JPEG.

```python
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
```
Ba công cụ của scikit-learn: vẽ confusion matrix, tính accuracy, tính precision/recall/F1. Dùng thư viện chuẩn thay vì tự viết để tránh sai công thức.

```python
from torch import nn
```
`nn` chứa các lớp mạng (Conv2d, Linear, BatchNorm…) và hàm loss.

```python
from torch.utils.data import DataLoader, Dataset
```
`Dataset` mô tả cách lấy một mẫu dữ liệu; `DataLoader` gom mẫu thành batch, xáo trộn và đọc song song.

```python
from torchvision import transforms as T
```
Các phép biến đổi ảnh (resize, crop, lật…). Đặt tên tắt `T` cho gọn.

```python
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
```
Hàm tạo mạng EfficientNet-B0 và danh sách bộ trọng số pretrained của nó.

```python
ROOT = Path("/content/drive/MyDrive/Deep Learning - RealWaste")
```
Thư mục gốc của dự án trên Drive. Mọi đường dẫn khác ghép từ đây, nên đổi máy chỉ cần sửa một dòng.

```python
DATA_DIR = ROOT / "01_Dataset/RealWaste"
```
Thư mục ảnh, bên trong có 9 thư mục con, mỗi thư mục một lớp.

```python
SPLIT_FILE = ROOT / "02_Splits/split_grouped_v2.csv"
```
File chia train/val/test đã chốt. Mỗi dòng là một ảnh: đường dẫn, tên lớp, tập.

```python
CKPT_DIR = ROOT / "03_Checkpoints"
RESULT_DIR = ROOT / "04_Results/experiments"
```
Nơi lưu trọng số model (checkpoint) và nơi lưu kết quả (lịch sử train, metric).

```python
SEEDS = [42, 43, 44]
```
Ba seed cho mỗi model E1–E3. Một lần train có yếu tố may rủi; 3 lần cho phép báo cáo trung bình ± độ lệch chuẩn.

```python
BATCH_SIZE = 32
```
Mỗi bước cập nhật trọng số dùng 32 ảnh. Đây là giá trị phổ biến, vừa bộ nhớ GPU T4.

```python
PATIENCE = 5
```
5 epoch liền không cải thiện macro-F1 validation thì dừng train (early stopping), tránh train thừa và overfit.

```python
EPOCHS = {"E1": 40, "E2": 40, "E3_stageA": 10, "E3_stageB": 15}
```
Số epoch tối đa của từng model. E3 có hai giai đoạn nên có hai mục. E3 cần ít epoch hơn vì đã học sẵn trên ImageNet.

```python
NUM_WORKERS = 2
```
Số tiến trình phụ đọc và biến đổi ảnh song song với GPU. Colab cho 2 CPU nên đặt 2.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Dùng GPU (`cuda`) nếu có, không thì CPU. Mọi tensor và model sau này được chuyển tới `device`.

```python
AMP = device.type == "cuda"
```
AMP (mixed precision) tính một phần bằng số thực 16 bit để nhanh hơn. Chỉ bật trên GPU, vì trên CPU không có lợi.

---

## Cell 4: đếm ảnh

```python
counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(DATA_DIR.iterdir()) if d.is_dir()}
```
Dict comprehension, đọc từ phải sang trái:
- `DATA_DIR.iterdir()`: liệt kê mọi thứ trong thư mục ảnh.
- `sorted(...)`: sắp theo tên cho thứ tự cố định.
- `if d.is_dir()`: chỉ lấy thư mục, tức các lớp.
- `d.glob("*.jpg")`: các file đuôi `.jpg` trong thư mục đó; `list(...)` rồi `len(...)` là số ảnh.
- Kết quả là dict `{tên lớp: số ảnh}`.

```python
print(counts, "total:", sum(counts.values()))
```
In số ảnh mỗi lớp và tổng (4,752).

```python
pd.Series(counts).sort_values().plot.barh(title="Images per class")
```
Đổi dict thành Series của pandas, sắp tăng dần, vẽ biểu đồ cột ngang. Nhìn biểu đồ thấy ngay dữ liệu lệch lớp (Plastic 921, Textile Trash 318).

```python
plt.show()
```
Hiển thị biểu đồ.

---

## Cell 6: chia dữ liệu theo khối

```python
def grouped_split(df, block_size=10, val_ratio=0.15, test_ratio=0.15, seed=42):
```
Hàm chia dữ liệu. Tham số mặc định đúng với thiết kế: khối 10 số, 15 % validation, 15 % test, seed 42.

```python
    df = df.sort_values("filepath").reset_index(drop=True)
```
Sắp theo đường dẫn để kết quả không phụ thuộc thứ tự dòng đầu vào. `reset_index(drop=True)` đánh lại chỉ số 0, 1, 2… và bỏ chỉ số cũ.

```python
    number = df["filepath"].str.extract(r"_(\d+)\.[A-Za-z]+$", expand=False).astype(int)
```
Lấy số cuối tên file bằng biểu thức chính quy (regex):
- `_` là dấu gạch dưới;
- `(\d+)` là một hoặc nhiều chữ số, và đây là phần được lấy ra;
- `\.[A-Za-z]+$` là dấu chấm và đuôi file ở cuối chuỗi.

Ví dụ `.../Plastic_646.jpg` → `"646"`. `.astype(int)` đổi chuỗi thành số.

```python
    df["block"] = number // block_size
```
`//` là chia lấy phần nguyên: 646 // 10 = 64. Ảnh 640–649 cùng khối 64. Vì ảnh chụp lại cùng một vật có số liền nhau, chúng gần như luôn rơi vào cùng khối.

```python
    rng = np.random.default_rng(seed)
```
Tạo bộ sinh số ngẫu nhiên riêng với seed 42: cùng seed thì lần nào chạy cũng xáo ra cùng một thứ tự.

```python
    split_of = {}
```
Dict rỗng, sẽ ghi `(lớp, khối) → "train"/"val"/"test"`.

```python
    for cls in sorted(df["class_name"].unique()):
```
Duyệt từng lớp, theo thứ tự tên. Chia riêng từng lớp (phân tầng, stratified), để mỗi lớp đều có tỉ lệ khoảng 70/15/15.

```python
        blocks = rng.permutation(np.sort(df.loc[df["class_name"] == cls, "block"].unique()))
```
- `df.loc[df["class_name"] == cls, "block"]`: các số khối của lớp này;
- `.unique()`: bỏ trùng;
- `np.sort`: sắp tăng dần, để đầu vào của phép xáo luôn giống nhau;
- `rng.permutation`: xáo trộn ngẫu nhiên (theo seed).

```python
        n_test, n_val = round(test_ratio * len(blocks)), round(val_ratio * len(blocks))
```
Số khối dành cho test và validation: 15 % số khối, làm tròn.

```python
        for i, b in enumerate(blocks):
```
Duyệt các khối đã xáo; `i` là vị trí (0, 1, 2…), `b` là số khối.

```python
            split_of[cls, b] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
```
`n_test` khối đầu → test, `n_val` khối tiếp → validation, còn lại → train. Cả khối đi chung một tập.

```python
    df["split"] = [split_of[key] for key in zip(df["class_name"], df["block"])]
```
Với từng ảnh, ghép `(lớp, khối)` của nó bằng `zip` rồi tra trong `split_of` để biết ảnh thuộc tập nào.

```python
    return df
```
Trả bảng đã có cột `split`.

```python
split_df = pd.read_csv(SPLIT_FILE)
```
Đọc file split thành DataFrame.

```python
again = grouped_split(split_df[["filepath", "class_name"]]).set_index("filepath")["split"]
```
Chạy lại hàm chia chỉ với hai cột đầu vào (đường dẫn, lớp). Kết quả lấy cột `split`, đánh chỉ số bằng đường dẫn để so từng ảnh.

```python
print("generated split equals the saved one:", again.eq(split_df.set_index("filepath")["split"]).all())
```
`.eq(...)` so từng ảnh giữa split vừa tạo và split đã lưu; `.all()` là True nếu tất cả giống. In `True` chứng minh file split mà mọi model dùng chính là sản phẩm của hàm này, không ai sửa tay.

```python
print(split_df["split"].value_counts())
```
Đếm số ảnh mỗi tập: train 3,326, val 719, test 707.

---

## Cell 8: pipeline dữ liệu

```python
mapping = pd.read_csv(ROOT / "02_Splits/class_mapping.csv")
```
File gán mỗi lớp một số từ 0 đến 8. Model chỉ hiểu nhãn là số.

```python
classes = mapping.sort_values("class_index")["class_name"].tolist()
```
Danh sách tên lớp theo đúng thứ tự số. Ví dụ `classes[0]` là `"Cardboard"`; dùng khi đổi số dự đoán ngược lại thành tên.

```python
split_df["label"] = split_df["class_name"].map(dict(zip(mapping["class_name"], mapping["class_index"])))
```
`dict(zip(...))` tạo dict `{tên lớp: số}`; `.map` thay tên lớp của từng ảnh bằng số tương ứng → cột `label`.

```python
split_df["image_path"] = [str(DATA_DIR / c / Path(f).name) for c, f in zip(split_df["class_name"], split_df["filepath"])]
```
Đường dẫn thật tới ảnh = thư mục ảnh / tên lớp / tên file. `Path(f).name` lấy phần tên file (ví dụ `Plastic_646.jpg`) từ đường dẫn đã ghi trong file split. Làm vậy để notebook chạy được dù Drive gắn ở chỗ khác.

```python
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
```
Trung bình và độ lệch chuẩn của ImageNet cho ba kênh R, G, B. EfficientNet-B0 được train với chuẩn hóa này nên bắt buộc phải dùng; hai model tự xây cũng dùng chung để tiền xử lý giống hệt nhau.

```python
train_tf = T.Compose([
```
`Compose` xếp các phép biến đổi thành chuỗi, áp lần lượt. Chuỗi này chỉ dùng cho tập train.

```python
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
```
Cắt ngẫu nhiên một vùng chiếm 80–100 % diện tích ảnh rồi resize về 224×224. Model thấy vật ở vị trí và tỉ lệ hơi khác mỗi lần. Không cắt nhỏ hơn 80 % để không mất vật.

```python
    T.RandomHorizontalFlip(0.5),
```
Lật ngang với xác suất 50 %. Rác lật ngang vẫn là rác cùng loại.

```python
    T.RandomRotation(10),
```
Xoay ngẫu nhiên trong khoảng −10° đến +10°. Góc nhỏ để ảnh vẫn tự nhiên.

```python
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
```
Thay đổi ngẫu nhiên độ sáng, tương phản, độ bão hòa tối đa ±20 %, mô phỏng ánh sáng khác nhau. `hue=0` giữ nguyên sắc màu, vì màu giúp phân biệt chất liệu (bìa nâu, rau xanh).

```python
    T.ToTensor(),
```
Ảnh PIL (cao × rộng × 3, giá trị 0–255) → tensor PyTorch (3 × cao × rộng, giá trị 0–1). PyTorch cần kênh đứng đầu.

```python
    T.Normalize(MEAN, STD),
])
```
Với mỗi kênh: (giá trị − mean) / std. Đưa dữ liệu về quanh 0, giúp train ổn định và khớp với cách EfficientNet được train.

```python
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
```
Cho validation và test: chỉ resize, đổi tensor, chuẩn hóa. Không có gì ngẫu nhiên, để chấm điểm lần nào cũng ra cùng kết quả. Ảnh gốc vuông 524×524 nên resize không làm méo.

```python
class WasteDataset(Dataset):
```
Lớp dữ liệu riêng, kế thừa `Dataset` của PyTorch. DataLoader yêu cầu có `__len__` và `__getitem__`.

```python
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
```
Lưu bảng ảnh (đánh lại chỉ số từ 0, để `iloc[i]` lấy đúng ảnh thứ i) và phép biến đổi sẽ dùng.

```python
    def __len__(self):
        return len(self.df)
```
Số mẫu trong tập. DataLoader dùng để biết có bao nhiêu batch.

```python
    def __getitem__(self, i):
        row = self.df.iloc[i]
```
DataLoader gọi hàm này với chỉ số `i`. `iloc[i]` lấy dòng thứ i của bảng.

```python
        image = Image.open(row.image_path).convert("RGB")
```
Mở ảnh và chắc chắn có 3 kênh màu.

```python
        return self.transform(image), row.label, row.filepath
```
Trả ba thứ: ảnh đã biến đổi, nhãn số, đường dẫn. Đường dẫn dùng để biết ảnh nào bị đoán sai khi phân tích lỗi.

```python
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```
Đặt seed cho cả bốn nguồn ngẫu nhiên: Python, numpy, PyTorch trên CPU, PyTorch trên GPU. Nhờ vậy khởi tạo trọng số, dropout và augmentation lặp lại được với cùng seed.

```python
def seed_worker(worker_id):
    # every loader worker gets its own random state, derived from the loader's seed
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
```
Mỗi worker đọc ảnh là một tiến trình riêng. PyTorch tự cho mỗi worker một seed torch khác nhau (suy ra từ seed của loader), lấy bằng `torch.initial_seed()`. Hàm này gán seed đó cho numpy và random trong worker. `% 2**32` vì numpy chỉ nhận seed nhỏ hơn 2³². Thiếu hàm này, các worker có thể sinh augmentation giống nhau hoặc không lặp lại được. `worker_id` không dùng tới nhưng DataLoader bắt buộc truyền vào.

```python
def make_loader(subset, transform, seed, shuffle=False):
```
Hàm tạo DataLoader cho một tập (`"train"`, `"val"` hoặc `"test"`).

```python
    data = WasteDataset(split_df[split_df["split"] == subset], transform)
```
Lọc các ảnh thuộc tập đó và bọc vào `WasteDataset`.

```python
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      persistent_workers=NUM_WORKERS > 0, pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
```
- `batch_size=32`: gom 32 ảnh một batch.
- `shuffle`: chỉ True cho train. Xáo mỗi epoch để model không học theo thứ tự; validation và test không cần xáo.
- `num_workers=2`: đọc ảnh song song.
- `persistent_workers`: giữ worker sống giữa các epoch, khỏi khởi động lại (chỉ hợp lệ khi có worker).
- `pin_memory`: đặt dữ liệu ở vùng nhớ chép sang GPU nhanh hơn.
- `worker_init_fn=seed_worker`: chạy hàm seed ở trên trong mỗi worker.
- `generator=...manual_seed(seed)`: thứ tự xáo và seed của worker lặp lại được với cùng seed.

---

## Cell 10: ba model

### SimpleCNN (E1)

```python
class SimpleCNN(nn.Module):
```
Mọi model PyTorch kế thừa `nn.Module`.

```python
    def __init__(self, num_classes=9, dropout=0.3, widths=(32, 64, 128)):
        super().__init__()
```
Tham số: 9 lớp, dropout 30 %, số kênh của 3 khối. `super().__init__()` khởi tạo phần `nn.Module`, bắt buộc để PyTorch theo dõi được các lớp con và tham số.

```python
        layers, c_in = [], 3
```
Danh sách lớp, sẽ thêm dần. `c_in = 3` là số kênh đầu vào (ảnh RGB).

```python
        for c_out in widths:
```
Mỗi vòng tạo một khối với `c_out` = 32, rồi 64, rồi 128 kênh.

```python
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
```
Một khối gồm bốn lớp:
- `Conv2d(c_in, c_out, 3, padding=1)`: `c_out` bộ lọc 3×3 trượt trên ảnh. Padding 1 thêm viền 1 điểm để đầu ra giữ nguyên kích thước. `bias=False` vì BatchNorm ngay sau đó trừ trung bình (bias mất tác dụng) và có tham số dịch riêng.
- `BatchNorm2d`: chuẩn hóa từng kênh theo batch. Train ổn định hơn và chịu được learning rate lớn hơn.
- `ReLU(inplace=True)`: max(0, x), tạo tính phi tuyến. `inplace` ghi đè luôn lên tensor cho đỡ tốn bộ nhớ.
- `MaxPool2d(2)`: lấy giá trị lớn nhất trong mỗi ô 2×2. Kích thước giảm một nửa, đỡ tính toán, ít nhạy với vị trí lệch nhỏ.

```python
            c_in = c_out
```
Số kênh ra của khối này là số kênh vào của khối sau.

```python
        self.features = nn.Sequential(*layers)
```
Ghép mọi lớp thành một chuỗi chạy lần lượt. `*layers` mở danh sách thành các đối số.

```python
        self.pool = nn.AdaptiveAvgPool2d(1)
```
Global average pooling: lấy trung bình mỗi feature map thành 1 số, bất kể kích thước. Đầu ra 128 số. Không có trọng số, nên ít overfit hơn flatten.

```python
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in, 128), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))
```
Phần phân loại:
- `Flatten`: (batch, 128, 1, 1) → (batch, 128).
- `Linear(128, 128)`: lớp kết nối đầy đủ.
- `ReLU`: phi tuyến.
- `Dropout(0.3)`: khi train, tắt ngẫu nhiên 30 % neuron để chống overfit.
- `Linear(128, 9)`: ra 9 điểm số (logits), mỗi lớp một số.

```python
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
```
Đường đi của ảnh: features → pool → classifier. Không có softmax ở đây vì `CrossEntropyLoss` đã tự làm.

### MultiScaleBlock

```python
class MultiScaleBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
```
Một khối đa tỉ lệ, vào `c_in` kênh, ra `c_out` kênh.

```python
        c1 = c5 = c_out // 4
```
Nhánh 1×1 và nhánh 5×5 mỗi nhánh lấy 1/4 số kênh ra.

```python
        self.branch1 = nn.Conv2d(c_in, c1, 1, bias=False)
```
Nhánh 1×1: nhìn từng điểm, trộn thông tin giữa các kênh. Không cần padding.

```python
        self.branch3 = nn.Conv2d(c_in, c_out - c1 - c5, 3, padding=1, bias=False)
```
Nhánh 3×3: chi tiết vừa. Lấy phần kênh còn lại (1/2), viết `c_out - c1 - c5` để tổng luôn đúng `c_out`. Padding 1 giữ kích thước.

```python
        self.branch5 = nn.Conv2d(c_in, c5, 5, padding=2, bias=False)
```
Nhánh 5×5: vùng rộng hơn, bắt cấu trúc lớn. Padding 2 giữ kích thước, để ba nhánh có cùng chiều cao và rộng mà ghép được.

```python
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
```
BatchNorm và ReLU áp lên kết quả đã ghép.

```python
    def forward(self, x):
        return self.act(self.bn(torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)))
```
Ba nhánh xử lý **cùng một đầu vào** `x`. `torch.cat(..., dim=1)` ghép theo chiều kênh (chiều 0 là batch, chiều 1 là kênh). Sau đó BatchNorm rồi ReLU. Ý tưởng lấy từ GoogLeNet/Inception (slide 5.2).

### MultiScaleCNN (E2)

```python
class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes=9, dropout=0.3, stem_width=32, widths=(64, 128, 256, 256)):
        super().__init__()
```
Số kênh: stem 32, bốn khối 64, 128, 256, 256.

```python
        layers = [nn.Conv2d(3, stem_width, 3, padding=1, bias=False), nn.BatchNorm2d(stem_width),
                  nn.ReLU(inplace=True), nn.MaxPool2d(2)]
```
"Stem" (phần đầu): một conv 3×3 thường giảm ảnh còn 112×112 trước khi vào các khối đa tỉ lệ, cho đỡ tốn tính toán.

```python
        c_in = stem_width
        for c_out in widths:
            layers += [MultiScaleBlock(c_in, c_out), nn.MaxPool2d(2)]
            c_in = c_out
```
Bốn khối đa tỉ lệ, mỗi khối kèm MaxPool. Kích thước 112 → 56 → 28 → 14 → 7.

```python
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in, 128), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
```
Giống E1: GAP rồi head. Ở đây `c_in` = 256 nên `Linear(256, 128)`.

### EfficientNetB0Transfer (E3)

```python
class EfficientNetB0Transfer(nn.Module):
    def __init__(self, num_classes=9, dropout=0.2, pretrained=False):
        super().__init__()
```
`pretrained=True` khi train (cần trọng số ImageNet); `False` khi chấm test (trọng số lấy từ checkpoint của nhóm, khỏi tải về).

```python
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
```
Tạo EfficientNet-B0, có hoặc không có trọng số ImageNet.

```python
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
```
Thay head gốc (1000 lớp ImageNet) bằng head mới 9 lớp. 1280 là số đặc trưng mà backbone EfficientNet-B0 đưa ra.

```python
        self.trainable_blocks = len(self.net.features)
```
`features` có 9 khối. Mặc định coi như cả 9 khối được train (chưa đóng băng gì).

```python
    def set_trainable_blocks(self, n):
        # freeze every feature block except the last n; the head always trains
        self.trainable_blocks = n
```
Ghi lại số khối cuối được train.

```python
        for i, block in enumerate(self.net.features):
            block.requires_grad_(i >= len(self.net.features) - n)
```
Khối có chỉ số `i ≥ 9 − n` được tính gradient (được train); các khối trước bị đóng băng. n = 0: đóng băng cả 9 (giai đoạn A, chỉ train head). n = 3: train khối 6, 7, 8 (giai đoạn B). Head không nằm trong `features` nên luôn được train.

```python
        self.train(self.training)
```
`self.training` là cờ đang ở chế độ train hay eval. Gọi lại `train` với đúng chế độ hiện tại để áp quy tắc ở hàm dưới ngay sau khi đổi số khối.

```python
    def train(self, mode=True):
        # frozen blocks stay in eval mode, so their batch norm keeps the ImageNet statistics
        super().train(mode)
```
Viết lại hàm `train` có sẵn. Trước hết đặt chế độ cho toàn model như bình thường.

```python
        for block in self.net.features[:len(self.net.features) - self.trainable_blocks]:
            block.eval()
```
Ép các khối bị đóng băng về chế độ eval. Lý do: `model.train()` được gọi đầu mỗi epoch và sẽ bật BatchNorm cập nhật thống kê trung bình/phương sai theo dữ liệu mới. `requires_grad=False` không ngăn được việc này. Đóng băng "thật" nghĩa là giữ nguyên thống kê ImageNet, và lớp drop ngẫu nhiên bên trong khối cũng phải tắt.

```python
        return self
```
Theo quy ước của PyTorch, `train()` trả về chính model.

```python
    def forward(self, x):
        return self.net(x)
```
Đi qua toàn bộ mạng.

```python
MODELS = {"E1": SimpleCNN, "E2": MultiScaleCNN, "E3": EfficientNetB0Transfer}
```
Bảng tra tên thí nghiệm → lớp model, để các vòng lặp phía sau tạo đúng model theo tên.

```python
for name, model_class in MODELS.items():
    print(name, f"{sum(p.numel() for p in model_class().parameters()):,} parameters")
```
Tạo thử mỗi model, đếm tổng số tham số (`numel` = số phần tử của mỗi tensor trọng số) và in có dấu phẩy ngăn cách. Dùng để đối chiếu với report: 111,145 / 1,230,377 / 4,019,077.

---

## Cell 12: metric, dự đoán, vòng train

### `macro_metrics`

```python
def macro_metrics(y_true, y_pred):
```
Nhận nhãn thật và nhãn dự đoán, trả 4 chỉ số.

```python
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)),
                                                  average="macro", zero_division=0)
```
- Tính precision, recall, F1 cho từng lớp rồi lấy trung bình cộng 9 lớp (`average="macro"`). Mỗi lớp nặng như nhau, quan trọng vì dữ liệu lệch lớp.
- `labels=range(9)`: luôn tính đủ 9 lớp.
- `zero_division=0`: lớp không có dự đoán nào thì precision = 0 thay vì báo lỗi.
- Giá trị thứ tư (support, số mẫu mỗi lớp) không dùng nên gán `_`.

```python
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f1)}
```
Trả dict 4 chỉ số. `float()` đổi kiểu numpy sang số Python thường cho dễ in và lưu.

### `predict`

```python
@torch.no_grad()
```
Decorator: trong hàm này PyTorch không ghi lại phép tính để tính gradient. Nhanh hơn và đỡ bộ nhớ, vì dự đoán thì không cần học.

```python
def predict(model, loader, criterion=None):
    model.eval()
```
Chế độ eval: Dropout tắt; BatchNorm dùng thống kê đã học thay vì của batch hiện tại. Kết quả ổn định, không ngẫu nhiên.

```python
    probs, labels, paths, loss = [], [], [], 0.0
```
Danh sách gom kết quả của mọi batch, và tổng loss.

```python
    for x, y, p in loader:
        x, y = x.to(device), y.to(device)
```
Lấy từng batch (ảnh, nhãn, đường dẫn) và chuyển ảnh, nhãn lên GPU.

```python
        with torch.autocast(device.type, enabled=AMP):
            logits = model(x).float()
```
Chạy model trong chế độ mixed precision (nhanh hơn trên GPU). `.float()` đưa kết quả về 32 bit trước khi tính softmax và loss, cho chính xác.

```python
        if criterion is not None:
            loss += criterion(logits, y).item() * len(y)
```
Nếu có truyền hàm loss (khi tính validation loss): loss trung bình của batch × số ảnh trong batch = tổng loss của batch. `.item()` lấy số Python từ tensor.

```python
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(y.cpu())
        paths += p
```
Softmax đổi logits thành xác suất (tổng bằng 1 trên 9 lớp). Chuyển về CPU và gom lại cùng nhãn và đường dẫn.

```python
    probs, y_true = torch.cat(probs).numpy(), torch.cat(labels).numpy()
```
Ghép kết quả mọi batch thành một mảng numpy.

```python
    return {"y_true": y_true, "y_pred": probs.argmax(1), "confidence": probs.max(1), "paths": paths,
            "loss": loss / len(y_true)}
```
- `argmax(1)`: lớp có xác suất cao nhất là dự đoán.
- `max(1)`: xác suất đó, tức độ tự tin.
- Loss chia cho tổng số ảnh → loss trung bình.

### `fit`

```python
def fit(model, train_loader, val_loader, lr, max_epochs, ckpt_dir, name, history_path):
```
Train một model. `name` dùng đặt tên file checkpoint (ví dụ `E1_best.pt`).

```python
    ckpt_dir.mkdir(parents=True, exist_ok=True)
```
Tạo thư mục checkpoint nếu chưa có. `parents=True` tạo cả thư mục cha; `exist_ok=True` không báo lỗi nếu đã có.

```python
    best_path, last_path = ckpt_dir / f"{name}_best.pt", ckpt_dir / f"{name}_last.pt"
```
Hai file: `_best` là trọng số epoch tốt nhất (dùng để chấm test); `_last` là toàn bộ trạng thái epoch gần nhất (dùng để train tiếp nếu bị ngắt).

```python
    criterion = nn.CrossEntropyLoss()
```
Hàm loss cho phân loại nhiều lớp. Bên trong tự làm log-softmax, nên model chỉ cần trả logits. Không đánh trọng số lớp (đã cân nhắc và quyết định không cần).

```python
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
```
AdamW: tự điều chỉnh bước học cho từng trọng số, cộng weight decay (kéo trọng số về 0 để chống overfit) tách riêng khỏi gradient. Chỉ đưa vào các tham số được train; với E3 thì các khối đóng băng bị loại.

```python
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
```
Giảm learning rate khi bị "chững": macro-F1 validation (`mode="max"`: càng cao càng tốt) không tăng 2 epoch thì nhân lr với 0.5. Lr nhỏ giúp tinh chỉnh khi đã gần tối ưu.

```python
    scaler = torch.amp.GradScaler(device.type, enabled=AMP)
```
Khi tính bằng số 16 bit, gradient rất nhỏ có thể bị làm tròn thành 0. GradScaler nhân loss lên trước khi lan truyền ngược rồi chia lại trước khi cập nhật trọng số. Tắt khi không dùng AMP.

```python
    history, best_f1, best_epoch, stale, start = [], -1.0, 0, 0, 1
```
Giá trị ban đầu: lịch sử rỗng; best = −1 để epoch đầu chắc chắn được lưu; `stale` đếm số epoch liền không cải thiện; bắt đầu từ epoch 1.

```python
    if last_path.exists():  # continue an interrupted run, or skip a finished one
        state = torch.load(last_path, map_location=device)
```
Nếu đã có `_last.pt` thì đọc lên. `map_location=device` đặt tensor thẳng vào GPU/CPU đang dùng.

```python
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
```
Khôi phục đủ bốn thứ: trọng số, trạng thái AdamW (các trung bình động), learning rate hiện tại của scheduler, và hệ số của scaler. Thiếu một thứ thì train tiếp sẽ khác với train liền một mạch.

```python
        history, best_f1, best_epoch = state["history"], state["best_val_macro_f1"], state["best_epoch"]
        stale, start = state["epochs_without_improvement"], state["epoch"] + 1
```
Khôi phục lịch sử, điểm tốt nhất, bộ đếm; bắt đầu từ epoch kế tiếp.

```python
        if stale >= PATIENCE or state["epoch"] >= max_epochs:
            print(f"{name}: already finished at epoch {state['epoch']}, best epoch {best_epoch}")
            return pd.DataFrame(history)
```
Run đã dừng sớm hoặc đã đủ epoch thì không train nữa, trả lịch sử. Đây là lý do chạy lại notebook không train lại.

```python
    for epoch in range(start, max_epochs + 1):
        t0 = time.time()
        model.train()
```
Vòng epoch. Ghi giờ bắt đầu. `model.train()` bật Dropout và cho BatchNorm học thống kê (trừ khối đóng băng của E3).

```python
        loss_sum, correct = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
```
Duyệt từng batch train; đường dẫn không cần nên bỏ (`_`).

```python
            optimizer.zero_grad()
```
Xoá gradient của bước trước. PyTorch cộng dồn gradient nên phải xoá mỗi bước.

```python
            with torch.autocast(device.type, enabled=AMP):
                logits = model(x)
                loss = criterion(logits, y)
```
Lan truyền xuôi trong chế độ mixed precision và tính loss.

```python
            scaler.scale(loss).backward()
```
Nhân loss với hệ số scale rồi lan truyền ngược (backpropagation), tính gradient cho mọi trọng số đang train.

```python
            scaler.step(optimizer)
```
Chia gradient lại cho hệ số scale rồi cho AdamW cập nhật trọng số. Nếu gradient bị tràn số (inf/NaN) thì bỏ qua bước này.

```python
            scaler.update()
```
Điều chỉnh hệ số scale cho bước sau: tràn thì giảm, ổn định lâu thì tăng.

```python
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
```
Cộng dồn tổng loss và số ảnh đoán đúng trong batch, để tính loss và accuracy train của cả epoch.

```python
        n = len(train_loader.dataset)
```
Tổng số ảnh train (3,326).

```python
        val = predict(model, val_loader, criterion)
        m = macro_metrics(val["y_true"], val["y_pred"])
```
Cuối epoch, chấm trên validation: lấy loss và 4 chỉ số.

```python
        history.append({"epoch": epoch, "train_loss": round(loss_sum / n, 5), "train_accuracy": round(correct / n, 5),
                        "val_loss": round(val["loss"], 5), "val_accuracy": round(m["accuracy"], 5),
                        "val_macro_f1": round(m["macro_f1"], 5), "lr": optimizer.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
```
Ghi một dòng lịch sử, làm tròn 5 chữ số: loss và accuracy của train và validation, macro-F1 validation, learning rate hiện tại (`param_groups[0]["lr"]`), số giây của epoch. File này dùng để vẽ đường cong train.

```python
        if history[-1]["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = history[-1]["val_macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": m}, best_path)
```
Macro-F1 validation cao hơn mức tốt nhất thì cập nhật best, reset bộ đếm và lưu trọng số (`state_dict` là dict tên lớp → tensor trọng số). Chọn theo validation chứ không theo test, để test vẫn "chưa nhìn thấy".

```python
        else:
            stale += 1
```
Không cải thiện thì tăng bộ đếm.

```python
        scheduler.step(history[-1]["val_macro_f1"])
```
Báo điểm epoch này cho scheduler; nó tự quyết có giảm learning rate không.

```python
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "history": history, "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
                    "epochs_without_improvement": stale}, last_path)
```
Lưu toàn bộ trạng thái sau mỗi epoch. Colab bị ngắt thì chỉ mất epoch đang chạy dở.

```python
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
```
Tạo thư mục kết quả nếu chưa có (Drive đôi khi làm mất thư mục mới tạo), rồi ghi lịch sử ra CSV.

```python
        print(f"{name} epoch {epoch}: train loss {loss_sum / n:.4f}, val loss {val['loss']:.4f}, "
              f"val macro-F1 {m['macro_f1']:.4f}")
```
In tiến độ, 4 chữ số thập phân.

```python
        if stale >= PATIENCE:
            break
```
5 epoch liền không cải thiện thì dừng (early stopping).

```python
    return pd.DataFrame(history)
```
Trả lịch sử dạng bảng.

---

## Cell 13: chạy train

```python
def run_dirs(exp, seed):
    # seed 42 is the original run; the other seeds live in trials/seedNN
    sub = "" if seed == 42 else f"trials/seed{seed}"
    return CKPT_DIR / exp / sub, RESULT_DIR / exp / sub
```
Đường dẫn checkpoint và kết quả của một run. Seed 42 là lần chạy gốc, nằm thẳng trong `03_Checkpoints/E1`; seed 43, 44 nằm trong `trials/seed43`, `trials/seed44`. Ghép với chuỗi rỗng `""` thì đường dẫn không đổi.

```python
def train_run(exp, seed):
    ckpt, res = run_dirs(exp, seed)
```
Train một model với một seed.

```python
    train_loader = make_loader("train", train_tf, seed, shuffle=True)
```
Loader train dùng `train_tf` (có augmentation) và xáo trộn mỗi epoch.

```python
    val_loader = make_loader("val", eval_tf, seed)
```
Loader validation, không ngẫu nhiên.

```python
    seed_everything(seed)
```
Đặt seed **trước khi tạo model**, để trọng số khởi tạo ngẫu nhiên lặp lại được và khác nhau giữa các seed. Thứ tự "tạo loader rồi đặt seed rồi tạo model" giống hệt lúc train thật.

```python
    if exp != "E3":
        model = MODELS[exp]().to(device)
        fit(model, train_loader, val_loader, 1e-3, EPOCHS[exp], ckpt, exp, res / f"{exp}_history.csv")
        return
```
E1, E2: tạo model, đưa lên GPU, train một lần với learning rate 0.001.

```python
    # E3 stage A: frozen backbone, only the new head learns
    model = EfficientNetB0Transfer(pretrained=True).to(device)
    model.set_trainable_blocks(0)
```
E3 giai đoạn A: tải trọng số ImageNet, đóng băng cả 9 khối, chỉ train head (feature extraction).

```python
    a = fit(model, train_loader, val_loader, 1e-3, EPOCHS["E3_stageA"], ckpt, "E3_stageA", res / "E3_stageA_history.csv")
```
Train head 10 epoch với lr 0.001. Head mới khởi tạo ngẫu nhiên nên cần lr lớn.

```python
    # E3 stage B: start from the best stage-A weights and fine-tune the last 3 blocks with a smaller learning rate
    model.load_state_dict(torch.load(ckpt / "E3_stageA_best.pt", map_location=device)["model"])
```
Nạp trọng số tốt nhất của giai đoạn A (không phải trọng số epoch cuối).

```python
    model.set_trainable_blocks(3)
```
Mở thêm 3 khối cuối của backbone để fine-tune. Đây là các khối mang đặc trưng cấp cao, cần điều chỉnh cho ảnh rác. 6 khối đầu (cạnh, kết cấu chung) giữ nguyên.

```python
    b = fit(model, train_loader, val_loader, 1e-4, EPOCHS["E3_stageB"], ckpt, "E3_stageB", res / "E3_stageB_history.csv")
```
Fine-tune 15 epoch với lr nhỏ hơn 10 lần (0.0001), để chỉnh nhẹ chứ không phá trọng số đã học.

```python
    best_stage = "B" if b["val_macro_f1"].max() > a["val_macro_f1"].max() else "A"
```
Giai đoạn nào có macro-F1 validation cao hơn thì chọn; bằng nhau thì giữ A (đơn giản hơn).

```python
    if not (ckpt / "E3_best.pt").exists():
        shutil.copyfile(ckpt / f"E3_stage{best_stage}_best.pt", ckpt / "E3_best.pt")
```
Chép checkpoint của giai đoạn được chọn thành `E3_best.pt`, để E3 có cùng tên file như các model khác khi chấm test. Chỉ chép nếu chưa có, để không ghi đè kết quả đã chốt.

```python
for exp in ["E1", "E2", "E3"]:
    for seed in SEEDS:
        train_run(exp, seed)
```
Train tất cả: 3 model × 3 seed = 9 run. Khi chạy lại, mọi run in "already finished".

---

## Cell 14: vẽ đường cong train

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
```
Một hàng 3 biểu đồ, kích thước 16×4 inch.

```python
for ax, exp in zip(axes, ["E1", "E2", "E3"]):
    h = pd.read_csv(RESULT_DIR / exp / f"{exp}_history.csv")
```
Mỗi biểu đồ một model; đọc lịch sử train của lần chạy seed 42.

```python
    ax.plot(h["epoch"], h["train_loss"], label="train loss")
    ax.plot(h["epoch"], h["val_loss"], label="validation loss")
    ax.plot(h["epoch"], h["val_macro_f1"], label="validation macro-F1")
```
Ba đường theo epoch. Train loss giảm mà validation loss không giảm là dấu hiệu overfit.

```python
    ax.set(title=f"{exp}, seed 42", xlabel="epoch")
    ax.legend()
plt.show()
```
Tiêu đề, nhãn trục, chú thích, hiển thị.

---

## Cell 16: chấm trên tập test

```python
test_loader = make_loader("test", eval_tf, 42)
```
Loader 707 ảnh test, không augmentation, không xáo.

```python
rows, outputs = [], {}
```
`rows`: một dòng kết quả cho mỗi run. `outputs`: dự đoán của các run seed 42, dùng cho confusion matrix và phân tích lỗi.

```python
for exp in ["E1", "E2", "E3"]:
    for seed in SEEDS:
        ckpt, res = run_dirs(exp, seed)
```
Duyệt đủ 9 run.

```python
        state = torch.load(ckpt / f"{exp}_best.pt", map_location=device)
        model = MODELS[exp]().to(device)
        model.load_state_dict(state["model"])
```
Đọc checkpoint tốt nhất, tạo model rỗng cùng kiến trúc, nạp trọng số vào. Tên lớp trong model phải khớp với tên trong checkpoint, nếu không thì báo lỗi.

```python
        out = predict(model, test_loader)
```
Dự đoán toàn bộ tập test.

```python
        stored = res / f"{exp}_metrics.json" if seed == 42 else RESULT_DIR / exp / f"test_seeds/seed{seed}/{exp}_metrics.json"
```
Đường dẫn file kết quả mà notebook 09 đã lưu cho run này.

```python
        rows.append({"model": exp, "seed": seed, "val_macro_f1": state["val_metrics"]["macro_f1"],
                     **macro_metrics(out["y_true"], out["y_pred"]),
                     "stored_macro_f1": json.loads(stored.read_text())["macro_f1"] if stored.exists() else None})
```
Một dòng kết quả gồm:
- tên model, seed;
- macro-F1 validation lưu trong checkpoint (dùng để chọn model cho phân tích lỗi);
- 4 chỉ số test (`**` mở dict thành các cột);
- macro-F1 đã lưu trước đó, để so và chứng minh code này ra đúng số cũ. Không có file thì để trống.

```python
        if seed == 42:
            outputs[exp] = out
```
Giữ dự đoán của run seed 42.

```python
results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))
```
Bảng 9 dòng, làm tròn 4 chữ số.

```python
metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "val_macro_f1"]
results.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
```
Nhóm theo model, tính trung bình và độ lệch chuẩn qua 3 seed. `std` của pandas là độ lệch chuẩn mẫu (chia n − 1). Dòng cuối của ô nên notebook tự hiển thị bảng. Đây là bảng kết quả chính của report.

---

## Cell 17: F1 theo lớp và confusion matrix

```python
f1_per_class = {exp: precision_recall_fscore_support(o["y_true"], o["y_pred"], labels=range(len(classes)),
                                                     zero_division=0)[2] for exp, o in outputs.items()}
```
Không truyền `average` thì hàm trả kết quả từng lớp; `[2]` lấy phần F1. Ra dict model → mảng 9 F1.

```python
print(pd.DataFrame(f1_per_class, index=classes).round(3))
```
Bảng: hàng là lớp, cột là model.

```python
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, (exp, o) in zip(axes, outputs.items()):
```
Một hàng 3 ô cho 3 model. `zip` ghép mỗi ô với dự đoán của một model.

```python
    ConfusionMatrixDisplay.from_predictions(o["y_true"], o["y_pred"], labels=range(len(classes)),
                                            display_labels=classes, xticks_rotation=45, colorbar=False, ax=ax)
```
Vẽ confusion matrix: hàng là lớp thật, cột là lớp dự đoán, ô là số ảnh. Đường chéo là đoán đúng; ô ngoài chéo cho biết lớp nào hay bị nhầm với lớp nào. Tên lớp trục ngang xoay 45° cho khỏi đè nhau.

```python
    ax.set_title(f"{exp}, seed 42")
plt.tight_layout()
plt.show()
```
Tiêu đề, tự căn khoảng cách, hiển thị.

---

## Cell 19: phân tích lỗi

```python
best = results.groupby("model")["val_macro_f1"].mean().idxmax()
```
Model có macro-F1 **validation** trung bình cao nhất (`idxmax` trả tên). Chọn theo validation, không theo test, để không "chọn sau khi xem đáp án".

```python
o = outputs[best]
```
Dự đoán test (seed 42) của model đó.

```python
errors = pd.DataFrame({"path": o["paths"], "true": [classes[i] for i in o["y_true"]],
                       "predicted": [classes[i] for i in o["y_pred"]], "confidence": o["confidence"]})
```
Bảng mọi ảnh test: đường dẫn, tên lớp thật, tên lớp dự đoán (đổi số sang tên qua `classes`), độ tự tin.

```python
errors = errors[errors["true"] != errors["predicted"]]
```
Chỉ giữ ảnh bị đoán sai.

```python
pairs = errors.groupby(["true", "predicted"]).size().sort_values(ascending=False)
```
Đếm số ảnh của từng cặp (thật → dự đoán), sắp giảm dần: cặp nhầm nhiều nhất lên đầu.

```python
print(best, len(errors), "wrong test predictions")
print(pairs.head(10))
```
In tên model, số ảnh sai và 10 cặp nhầm nhiều nhất.

```python
examples = pd.concat([errors[(errors["true"] == t) & (errors["predicted"] == p)].head(4) for t, p in pairs.index[:6]])
```
Với 6 cặp nhầm nhiều nhất (`pairs.index` là các cặp `(thật, dự đoán)`), lấy tối đa 4 ảnh mỗi cặp, ghép thành một bảng.

```python
fig, axes = plt.subplots(-(-len(examples) // 6), 6, figsize=(16, 3 * -(-len(examples) // 6)), squeeze=False)
```
Lưới 6 cột. `-(-n // 6)` là chia làm tròn lên (ví dụ 20 ảnh → 4 hàng). `squeeze=False` giữ `axes` là mảng 2 chiều kể cả khi chỉ có 1 hàng.

```python
for ax in axes.flat:
    ax.axis("off")
```
Tắt trục ở mọi ô (ô thừa sẽ trống).

```python
for ax, r in zip(axes.flat, examples.itertuples()):
    ax.imshow(Image.open(DATA_DIR / r.true / Path(r.path).name))
    ax.set_title(f"{r.true} -> {r.predicted}\n{r.confidence:.2f}", fontsize=8)
```
Mỗi ô một ảnh lỗi. `itertuples()` duyệt từng dòng, truy cập cột bằng `r.true`… Tiêu đề: lớp thật → lớp dự đoán, xuống dòng, độ tự tin.

```python
plt.tight_layout()
plt.show()
```
Căn khoảng cách và hiển thị.

---

## Các câu giảng viên hay hỏi

1. **Vì sao dùng global average pooling thay vì flatten?** Flatten 7×7×256 ra 12,544 số; lớp dense sau đó cần hàng triệu trọng số và dễ overfit. GAP không có trọng số và chỉ còn 256 số.
2. **Vì sao conv có `bias=False`?** BatchNorm ngay sau trừ trung bình (làm mất bias) rồi cộng tham số dịch beta của nó. Bias của conv thành thừa.
3. **Vì sao model không có softmax?** `CrossEntropyLoss` đã gồm log-softmax. Softmax chỉ dùng khi cần xác suất, trong `predict`.
4. **Vì sao chọn theo macro-F1?** Lớp lệch nhau 2,9 lần. Accuracy có thể cao dù bỏ sót gần hết Textile Trash; macro-F1 cho mỗi lớp trọng số như nhau.
5. **Vì sao model tự xây cũng chuẩn hóa theo ImageNet?** Để tiền xử lý giống hệt nhau giữa các model. Đây là hằng số cố định, không ước lượng từ dữ liệu của nhóm nên không rò rỉ.
6. **Vì sao giai đoạn B dùng lr 1e-4?** Các khối pretrained đã tốt; lr lớn sẽ phá đặc trưng ImageNet, lr nhỏ chỉ chỉnh nhẹ.
7. **Vì sao phải viết lại `train()` cho EfficientNet?** `requires_grad=False` chỉ chặn gradient. BatchNorm vẫn cập nhật thống kê khi ở chế độ train, nên phải ép khối đóng băng về eval.
8. **Receptive field 22 và 154 tính thế nào?** Mỗi conv k×k mở rộng vùng nhìn thêm (k − 1) × bước nhảy hiện tại; mỗi pooling 2×2 thêm 1 × bước nhảy hiện tại rồi nhân bước nhảy lên 2. E1: 3 → 4 → 8 → 10 → 18 → 22.
9. **Early stopping và ReduceLROnPlateau khác nhau thế nào?** Cả hai theo dõi macro-F1 validation. Sau 2 epoch không tăng thì giảm lr một nửa; sau 5 epoch không tăng thì dừng hẳn.
10. **Vì sao train 3 seed?** Một lần train có yếu tố may rủi (khởi tạo, thứ tự batch, augmentation, dropout). 3 seed cho trung bình ± độ lệch chuẩn. Split giữ nguyên, nên SD chỉ đo dao động do train.
11. **Tập test dùng mấy lần?** Chỉ để chấm sau khi đã chọn xong model; không chọn hay sửa gì dựa trên điểm test. Chạy lại notebook chỉ tính lại cùng con số từ cùng trọng số.
12. **AMP là gì?** Tính bằng số 16 bit trên GPU cho nhanh. GradScaler chống gradient bị làm tròn về 0. Kết quả gần như không đổi.
13. **Colab bị ngắt giữa chừng thì sao?** `_last.pt` lưu sau mỗi epoch; chạy lại thì `fit` đọc file đó và train tiếp, chỉ mất epoch đang dở.
14. **Vì sao không tuning siêu tham số?** Không đủ thời gian. Nhóm dùng giá trị phổ biến, giống nhau cho mọi model, và nêu rõ đây là hạn chế trong report.
15. **Vì sao chia theo khối 10 số mà không chia ngẫu nhiên?** Cùng một vật được chụp lại với số file liền nhau. Chia ngẫu nhiên thì ảnh này vào train, ảnh kia vào test, model được "nhắc bài". Chia theo khối giữ các ảnh chụp lại trong cùng một tập.
16. **Làm sao nhóm phát hiện ảnh chụp lại?** Xem tận mắt 32 bộ ba ảnh có số file liền nhau, chọn ngẫu nhiên: 17 bộ có cùng một vật bị chụp lại, thường sau khi bị dịch chỗ hoặc lật. Với cách chia ngẫu nhiên, 4 trong 32 vật đó có ảnh ở cả train lẫn val/test; với cách chia theo khối chỉ còn 1.
