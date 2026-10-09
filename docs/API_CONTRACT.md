# Shared API Contract — Stage 0

Tài liệu này là hợp đồng ổn định giữa `src/common/` và ba task. Không thay đổi
signature hoặc ý nghĩa của dữ liệu dùng chung trong branch task nếu chưa được cả
nhóm thống nhất.

## 1. Embedding API

Import:

```python
from src.common import (
    EmbeddingFormatError,
    EmbeddingStore,
    OutOfVocabularyError,
    load_embeddings,
)
```

### `load_embeddings(path, *, dtype=np.float32) -> EmbeddingStore`

Đọc UTF-8 nghiêm ngặt theo format:

```text
<number_of_words>
<dimension>
<word> <float_1> ... <float_dimension>
```

Validation:

- file phải tồn tại;
- hai header phải là số nguyên dương trên hai dòng riêng;
- số dòng vector phải đúng với header;
- mỗi dòng phải có đúng `dimension` giá trị;
- token không được trùng;
- mọi giá trị phải là số hữu hạn;
- dòng trống giữa các vector là lỗi.

Exception:

- `FileNotFoundError`: sai đường dẫn;
- `EmbeddingFormatError`: schema, UTF-8 hoặc dữ liệu số không hợp lệ;
- `TypeError`: `dtype` không phải floating-point.

### `EmbeddingStore`

| Thành phần | Kiểu trả về | Hành vi |
|---|---|---|
| `size` | `int` | số token |
| `dimension` | `int` | số chiều vector |
| `contains(word)` | `bool` | kiểm tra OOV |
| `word in embeddings` | `bool` | tương đương `contains` |
| `get_vector(word)` | `np.ndarray`, shape `(d,)` | view read-only; OOV thì raise |
| `get_vocabulary()` | `tuple[str, ...]` | cùng thứ tự với row matrix |
| `get_embedding_matrix()` | `np.ndarray`, shape `(n,d)` | shared matrix read-only |

Ví dụ:

```python
embeddings = load_embeddings("data/word2vec/W2V_150.txt")
if embeddings.contains("sinh_viên"):
    vector = embeddings.get_vector("sinh_viên")

try:
    embeddings.get_vector("token_không_tồn_tại")
except OutOfVocabularyError as error:
    print(error)
```

Không sửa trực tiếp matrix/vector. Task cần normalized matrix có thể tạo bản sao
riêng hoặc cache trong module của mình.

## 2. Dataset API

Import:

```python
from src.common import load_visim, load_vicon, load_training_pairs
```

### `load_visim(path) -> list[ViSimPair]`

`ViSimPair` là frozen dataclass:

```python
ViSimPair(
    word1: str,
    word2: str,
    pos: str,
    sim1: float,
    sim2: float,
    std: float,
)
```

Header bắt buộc: `Word1 Word2 POS Sim1 Sim2 STD` phân tách bằng tab. Loader
không normalize chữ, dấu tiếng Việt, dấu gạch dưới hoặc điểm số.

### `load_vicon(path) -> list[LabeledWordPair]`

Nhận một file hoặc thư mục ViCon chuẩn. Nếu nhận thư mục, loader đọc theo thứ tự
noun → verb → adjective và gán `pos` lần lượt `N`, `V`, `A`.

```python
LabeledWordPair(
    word1: str,
    word2: str,
    relation: Literal["SYN", "ANT"],
    pos: str | None,
)
```

Relation khác `SYN`/`ANT` là `DatasetFormatError`.

### `load_training_pairs(synonym_path, antonym_path) -> TrainingDataset`

Kết quả:

```python
dataset.pairs        # tuple[LabeledWordPair, ...]
dataset.diagnostics  # TrainingDiagnostics
```

Các dòng hợp lệ, kể cả duplicate hoặc conflict, được giữ nguyên theo thứ tự SYN
rồi ANT. Dòng không có đúng hai token không thể tạo pair nên được bỏ khỏi
`pairs`, nhưng lưu đầy đủ trong `malformed_rows` gồm source, line number, raw
value và reason.

`TrainingDiagnostics` cung cấp:

- `total_rows`, `valid_rows`;
- `label_counts`;
- `malformed_rows`;
- `duplicate_ordered_rows`: số bản sao dư của `(word1, word2, label)`;
- `duplicate_unordered_rows`: số bản sao dư khi coi `(w1,w2)` và `(w2,w1)` là
  cùng cặp, nhưng vẫn phân biệt label;
- `conflicting_unordered_pairs`: map cặp không thứ tự tới các label mâu thuẫn.

Không xem diagnostics như lệnh tự động xóa dữ liệu. Chính Task 3 phải chọn và
ghi chép protocol tiền xử lý.

## 3. Coverage API

```python
from src.common import get_coverage

report = get_coverage(pairs, embeddings)
```

`pairs` là iterable của object có hai thuộc tính string `word1`, `word2`.
`CoverageReport` gồm:

| Field/property | Ý nghĩa |
|---|---|
| `total_pairs` | tổng cặp đầu vào |
| `covered_pairs` | cả hai từ có vector |
| `oov_pairs` | thiếu ít nhất một vector |
| `pair_coverage` | `covered_pairs / total_pairs`, hoặc `0.0` nếu rỗng |
| `total_unique_words` | tổng token phân biệt |
| `covered_unique_words` | token phân biệt có vector |
| `word_coverage` | coverage trên unique token |
| `oov_unique_words` | tuple OOV đã sort |
| `oov_occurrences` | số lần mỗi OOV xuất hiện trong các vị trí cặp |

Hàm chỉ thống kê; không loại cặp và không sinh zero vector.

## 4. Error policy chung

- Evaluation dataset có schema sai: fail fast bằng `DatasetFormatError`.
- Training row hỏng riêng lẻ: ghi nhận trong diagnostics và tiếp tục.
- OOV khi truy xuất trực tiếp: `OutOfVocabularyError`.
- OOV trong dataset: `get_coverage` báo cáo; task quyết định chỉ tính metric trên
  covered subset và phải công bố mẫu số.
- Token được so khớp chính xác, có phân biệt Unicode và dấu gạch dưới; Stage 0
  không tự lowercase, sửa chính tả hoặc thay `_` bằng khoảng trắng.

## 5. CLI integration contract

Mỗi `src/taskN/cli.py` phải export:

```python
def run(args: argparse.Namespace) -> int:
    ...
```

- trả `0` nếu chạy thành công;
- raise exception có thông báo rõ nếu dữ liệu/config sai;
- không gọi `sys.exit()` bên trong module;
- chỉ ghi output vào đường dẫn `args.output` nếu task có output file;
- Stage 0 đang dùng `NotImplementedError` có chủ ý. Thành viên thay nội dung
  `run`, không đổi dispatcher trong `main.py` nếu không cần thêm tham số.
