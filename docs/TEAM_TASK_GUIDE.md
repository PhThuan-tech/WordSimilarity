# Team Task Guide

Stage 0 bảo đảm ba thành viên có thể bắt đầu song song. Mỗi người dùng
`src/common/`, chỉ sửa module task của mình, viết test riêng và tạo output riêng.

## Bảng phân công tổng quan

| Thành viên | Branch | Phạm vi source | Dataset chính | Kết quả chính |
|---|---|---|---|---|
| Member 1 | `feature/task1-similarity` | `src/task1/`, `tests/task1/` | ViSim-400 | cosine từng cặp, Pearson, Spearman, coverage |
| Member 2 | `feature/task2-nearest` | `src/task2/`, `tests/task2/` | Word2Vec | top-k cho một từ, score giảm dần |
| Member 3 | `feature/task3-classifier` | `src/task3/`, `tests/task3/` | train SYN/ANT, ViCon | predictions, Precision/Recall/F1, coverage và leakage report |

Tất cả cùng đọc [API_CONTRACT.md](API_CONTRACT.md). Không task nào được import
logic thuật toán từ task khác. Các thay đổi bắt buộc ở `src/common/` phải là PR
nhỏ, được thống nhất trước để tránh chặn hai thành viên còn lại.

---

# Member 1 — Cosine Similarity và ViSim-400

## 1. Mục tiêu

Tính cosine similarity từ pretrained Word2Vec cho từng cặp ViSim có đủ vector,
đánh giá mức tương quan với điểm người chấm và xuất prediction từng dòng.

Đầu ra tối thiểu:

- CSV chứa đủ 400 dòng và trạng thái covered/OOV;
- Pearson và Spearman trên đúng covered subset;
- coverage: `344/400` nếu dùng dữ liệu hiện tại;
- log ghi rõ dùng `Sim1` hay `Sim2` làm gold score. Nên báo cả hai hoặc chọn
  `Sim2` và giải thích vì đây chỉ là phép ánh xạ tuyến tính của cùng đánh giá.

## 2. Kiến thức cần nắm

Với hai vector khác zero:

\[
\cos(u,v)=\frac{u^Tv}{\lVert u\rVert_2\lVert v\rVert_2}
\]

Pearson đo quan hệ tuyến tính giữa prediction và human score. Spearman tính
Pearson trên thứ hạng, phù hợp để kiểm tra model có xếp thứ tự tương đồng đúng
hay không. Không đưa OOV hoặc `NaN` vào correlation. Nếu vector có norm bằng
zero, đánh dấu dòng không đánh giá được thay vì chia cho zero.

## 3. Chuẩn bị

```python
from src.common import get_coverage, load_embeddings, load_visim
```

Thư viện: NumPy; `scipy.stats.pearsonr`, `scipy.stats.spearmanr`; có thể dùng
Pandas chỉ để xuất CSV. Dataset:

- `data/word2vec/W2V_150.txt`
- `data/ViSim-400/Visim-400.txt`

## 4. Các bước thực hiện

1. Tạo hàm cosine thuần trong `src/task1/similarity.py`; kiểm tra shape, norm và
   trả kiểu `float`.
2. Load embedding và ViSim bằng API chung.
3. Gọi `get_coverage` trước khi lọc để lưu mẫu số đầy đủ.
4. Với từng pair, nếu covered thì tính cosine; nếu OOV thì để prediction trống
   và ghi tên từ thiếu.
5. Tính Pearson/Spearman chỉ trên prediction hữu hạn và gold tương ứng.
6. Xuất CSV và in/tạo JSON summary. Tạo parent output directory nếu chưa có.
7. Thay placeholder `run(args)` trong `src/task1/cli.py` để dùng
   `args.embeddings`, `args.dataset`, `args.output`.
8. Chạy unit test, integration test và CLI từ project root.

## 5. Phạm vi code

Được sửa/tạo: `src/task1/**`, `tests/task1/**`, output `results/task1_*` và tài
liệu Task 1 nếu cần. Không sửa `src/task2`, `src/task3`; không chép loader vào
task. Nếu cần sửa `main.py` hoặc common API, tách commit và báo cả nhóm.

## 6. Input/Output

Gợi ý API nội bộ:

```python
def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float: ...
def evaluate_visim(embeddings, pairs) -> tuple[list[dict], dict]: ...
```

CSV tối thiểu:

```text
word1,word2,pos,sim1,sim2,std,cosine,status,oov_words
```

Ví dụ CLI:

```bash
python main.py similarity --output results/task1_predictions.csv
```

## 7. Kiểm thử bắt buộc

- vector giống nhau → cosine gần `1`;
- vector vuông góc → `0`; ngược hướng → `-1`;
- symmetry: `cos(a,b) == cos(b,a)`;
- shape không khớp và zero norm có hành vi rõ;
- từ tiếng Việt có `_` truy xuất đúng;
- OOV không làm crash cả evaluation và không bị thay bằng zero;
- số prediction hữu hạn là 344 với dữ liệu hiện tại;
- Pearson/Spearman được so với phép tính nhỏ làm tay hoặc SciPy fixture;
- CSV giữ 400 hàng, kể cả OOV.

## 8. Kết quả bàn giao

- source và tests trong phạm vi Task 1;
- `results/task1_predictions.csv`;
- summary metric có sample count, Pearson, Spearman và coverage;
- đoạn README ngắn mô tả command và interpretation.

## 9. Definition of Done

- [ ] Không có thuật toán similarity nằm ngoài `src/task1/`.
- [ ] Unit tests và toàn bộ `python -m pytest -q` pass.
- [ ] CLI trả exit code 0 và tạo output tái lập được.
- [ ] 400 dòng được ghi, 344 dòng covered được xác nhận.
- [ ] Metric không chứa OOV/NaN và có ghi rõ gold column, `n`, coverage.
- [ ] Không sửa hoặc commit file dữ liệu gốc.

## 10. Git workflow

```bash
git switch -c feature/task1-similarity
git add src/task1 tests/task1
git commit -m "feat(task1): implement cosine similarity and ViSim evaluation"
git push -u origin feature/task1-similarity
```

Sau đó tạo PR vào integration branch theo [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

---

# Member 2 — K-nearest Words

## 1. Mục tiêu

Cho một token và số nguyên dương `k`, trả về tối đa `k` token khác có cosine
cao nhất, tận dụng phép tính vectorized trên toàn bộ ma trận 77.021 × 150.

## 2. Kiến thức cần nắm

Chuẩn hóa mỗi row embedding theo L2. Với query chuẩn hóa `q`, vector điểm là:

\[
s = E_{normalized}q
\]

Loại index của query. Có thể dùng `np.argpartition` để lấy candidate top-k rồi
sort chính xác phần nhỏ đó. Nếu ưu tiên đơn giản, `np.argsort` toàn bộ vẫn đúng
nhưng cần báo chi phí. Tie-break phải xác định, ví dụ score giảm dần rồi token
tăng dần, để test tái lập.

## 3. Chuẩn bị

```python
from src.common import OutOfVocabularyError, load_embeddings
```

Chỉ cần Word2Vec và NumPy. Member 2 tự viết phép cosine/matrix normalization
trong `src/task2/`; không import Task 1.

## 4. Các bước thực hiện

1. Tạo `src/task2/nearest.py` với class hoặc hàm index top-k.
2. Lấy vocabulary và matrix từ shared store; không sửa matrix read-only.
3. Chuẩn hóa vectorized, xử lý row norm zero an toàn; có thể cache normalized
   matrix một lần cho nhiều query.
4. Kiểm tra query có trong vocab trước khi tính.
5. Validate `k > 0`; nếu `k >= vocabulary size`, trả tối đa `size - 1` hoặc
   raise theo policy đã ghi rõ. Khuyến nghị clamp và thông báo.
6. Gán score query thành `-inf` hoặc loại index trước khi chọn top-k.
7. Sort kết quả theo contract ổn định.
8. Thay `src/task2/cli.py::run(args)` và in dạng dễ đọc.

## 5. Phạm vi code

Được sửa/tạo: `src/task2/**`, `tests/task2/**`, tài liệu/output Task 2. Không phụ
thuộc source Task 1. Không ghi normalized matrix đè lên shared embedding matrix.

## 6. Input/Output

Gợi ý:

```python
@dataclass(frozen=True)
class Neighbor:
    word: str
    similarity: float

def nearest_words(embeddings, word: str, k: int) -> list[Neighbor]: ...
```

CLI:

```bash
python main.py nearest --word sinh_viên --k 10
```

Output mỗi dòng gồm rank, token và cosine; không được chứa chính `sinh_viên`.

## 7. Kiểm thử bắt buộc

- toy matrix có top-k biết trước;
- query không xuất hiện trong output;
- score không tăng khi rank tăng;
- `k=1`, `k=10`, `k >= size`, `k=0`, `k<0`;
- OOV ném exception/thông báo rõ, không trả danh sách giả;
- token Unicode và `_`;
- row zero nếu có không sinh NaN;
- kết quả vectorized khớp brute-force trên fixture nhỏ;
- smoke test với `sinh_viên` trên dữ liệu thật.

## 8. Kết quả bàn giao

- source và tests Task 2;
- ví dụ output top-10 cho một số query hợp lệ;
- ghi chú thời gian load/index/query và policy cho tie, zero norm, k quá lớn.

## 9. Definition of Done

- [ ] Dùng shared loader và NumPy vectorized.
- [ ] Không import Task 1 và không trả query trong neighbors.
- [ ] OOV/k không hợp lệ được xử lý theo contract.
- [ ] Kết quả có thứ tự ổn định và khớp brute-force fixture.
- [ ] `python main.py nearest --word sinh_viên --k 10` thành công.
- [ ] Toàn bộ test pass, dữ liệu gốc không đổi.

## 10. Git workflow

```bash
git switch -c feature/task2-nearest
git add src/task2 tests/task2
git commit -m "feat(task2): implement vectorized nearest-word search"
git push -u origin feature/task2-nearest
```

---

# Member 3 — Synonym–Antonym Classification

## 1. Mục tiêu

Tạo feature từ hai Word2Vec, huấn luyện baseline Logistic Regression trên tập
SYN/ANT, dự đoán ViCon và báo Precision, Recall, F1, macro-F1 cùng coverage.
Không dùng dictionary/lookup nhãn từ train để dự đoán test.

## 2. Kiến thức cần nắm

Một baseline đối xứng theo thứ tự từ:

\[
x(w_1,w_2)=[|v_1-v_2|; v_1\odot v_2; \cos(v_1,v_2)]
\]

Không nên chỉ dùng cosine: synonym và antonym đều có thể gần nhau trong không
gian phân bố. Logistic Regression cần xử lý train imbalance, khuyến nghị
`class_weight="balanced"`. Báo per-class metrics và macro average. Fit scaler,
nếu dùng, chỉ trên train thông qua `Pipeline` để tránh leakage.

Đặc biệt: ViCon hiện chồng lặp train khi bỏ qua thứ tự. Đây là protocol của đề,
không phải clean generalization. Phải báo overlap; không được lookup nhãn. Một
thử nghiệm sạch chỉ là bổ sung và phải chia trước khi fit/preprocess.

## 3. Chuẩn bị

```python
from src.common import (
    get_coverage,
    load_embeddings,
    load_training_pairs,
    load_vicon,
)
```

Thư viện: NumPy, scikit-learn, Pandas cho output. Dữ liệu: hai file training và
thư mục `data/ViCon-400`.

## 4. Các bước thực hiện

1. Load training, in toàn bộ diagnostics trước mọi lựa chọn tiền xử lý.
2. Load ViCon và tính coverage riêng train/test.
3. Viết hàm audit unordered overlap và label conflict; lưu số liệu vào report.
4. Xác định policy cho malformed (đã tách khỏi pairs), duplicate và conflicts.
   Không âm thầm xóa. Ghi rõ số hàng trước/sau mỗi bước.
5. Tạo feature đối xứng chỉ cho cặp covered; trả thêm index để map prediction về
   row gốc.
6. Tạo sklearn `Pipeline`, Logistic Regression với seed cố định và class weight.
7. Fit trên train covered theo protocol đã công bố; không dùng test để chọn
   hyperparameter.
8. Dự đoán ViCon covered; OOV rows vẫn có mặt trong CSV với status rõ.
9. Tính confusion matrix, per-class Precision/Recall/F1, macro-F1 và accuracy
   trên covered test subset. Dùng `zero_division=0` và ghi `n`.
10. Nếu làm clean split bổ sung, canonicalize cặp trước khi split/group để bản
    đảo chiều không đi vào hai tập; giữ riêng kết quả này với required protocol.
11. Thay `src/task3/cli.py::run(args)`.

## 5. Phạm vi code

Được sửa/tạo: `src/task3/**`, `tests/task3/**`, `results/task3_*` và tài liệu
Task 3. Không sửa loader để lặng lẽ lọc dữ liệu cho riêng classifier.

## 6. Input/Output

Gợi ý API:

```python
def pair_features(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray: ...
def build_feature_matrix(pairs, embeddings) -> tuple[np.ndarray, np.ndarray]: ...
def audit_overlap(train_pairs, test_pairs) -> dict: ...
```

CSV tối thiểu:

```text
word1,word2,pos,gold,predicted,prob_syn,status,oov_words
```

CLI:

```bash
python main.py classify --output results/task3_predictions.csv
```

## 7. Kiểm thử bắt buộc

- feature shape đúng và hữu hạn;
- feature đối xứng khi đổi `w1`, `w2`;
- OOV không tạo zero feature;
- diagnostics thật: 13.558 valid, 4 malformed, 7 conflicting unordered pairs;
- class count 11.558 SYN/2.000 ANT trước filtering;
- ViCon 1.400 dòng, 700/700, covered 1.173;
- overlap audit nhận cả cặp đảo chiều;
- scaler/model chỉ fit train;
- metrics khớp confusion matrix toy;
- seed cố định cho kết quả tái lập;
- prediction CSV giữ đủ 1.400 dòng và đánh dấu OOV.

## 8. Kết quả bàn giao

- source/tests Task 3;
- prediction CSV đủ hàng;
- JSON/Markdown metrics có coverage, class distribution, diagnostics, confusion
  matrix, per-class và macro metrics;
- đoạn phân tích train/test overlap và giới hạn kết luận;
- nếu có MLP/clean split, để như thí nghiệm bổ sung, không thay baseline bắt buộc.

## 9. Definition of Done

- [ ] Không label lookup; model chỉ nhận embedding-derived features.
- [ ] Có baseline Logistic Regression và xử lý imbalance rõ ràng.
- [ ] Duplicate/conflict/overlap được audit và protocol được ghi lại.
- [ ] Metrics ghi positive-class convention, averaging, sample count, coverage.
- [ ] `python main.py classify` thành công và output tái lập.
- [ ] Toàn bộ test pass, không chỉnh sửa dữ liệu gốc.

## 10. Git workflow

```bash
git switch -c feature/task3-classifier
git add src/task3 tests/task3
git commit -m "feat(task3): implement synonym antonym classifier"
git push -u origin feature/task3-classifier
```

---

# Checklist nghiệm thu chung

- [ ] Mỗi task chỉ thay module sở hữu và test tương ứng.
- [ ] Ba CLI dùng chung đường dẫn mặc định và chạy từ repository root.
- [ ] Mọi output ghi vào `results/`, không ghi đè dataset.
- [ ] Mọi metric có coverage, sample count và OOV policy.
- [ ] `python -m pytest -q` pass sau khi merge cả ba branch.
- [ ] README cuối cùng chứa lệnh chạy và tóm tắt kết quả thật, không có metric giả.
- [ ] Ba PR đã được một thành viên khác review trước khi merge.
