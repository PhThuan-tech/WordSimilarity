# Vietnamese Word Similarity

Bài tập xử lý độ tương đồng từ tiếng Việt bằng pretrained Word2Vec, gồm ba task:

1. Tính cosine similarity và đánh giá trên ViSim-400.
2. Tìm `k` từ gần nhất với một từ cho trước.
3. Phân loại cặp từ đồng nghĩa (`SYN`) hoặc trái nghĩa (`ANT`).

Phần đọc và kiểm tra dữ liệu dùng chung cùng cả ba task đã được triển khai. Mỗi
task nằm trong module riêng, có CLI và unit test độc lập.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Chạy tại thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dataset và Word2Vec đã có sẵn trong `data/`, không cần tải thêm.

## Kiểm tra project

```powershell
python -m pytest -q
```

Toàn bộ test phải kết thúc với trạng thái `passed`.

Test suite sử dụng thư mục tạm riêng cho từng test nên không phụ thuộc vào thư
mục `%TEMP%` dùng chung của Windows.

## Cách chạy ba task

```powershell
# Task 1: Cosine similarity và ViSim-400
python main.py similarity

# Task 2: Tìm 10 từ gần nhất
python main.py nearest --word sinh_viên --k 10

# Task 3: Phân loại synonym–antonym
python main.py classify
```

Task 1 chạy được và tạo `results/task1_predictions.csv` cùng
`results/task1_predictions_summary.json`: 344/400 cặp covered (86%), Pearson
Sim2 = 0.4468197550, Spearman = 0.4077568888. Báo cáo có cả Sim1 và Sim2;
CSV giữ đủ 400 dòng, metric bỏ qua 56 cặp OOV. Xem [Task 1](docs/TASK1.md).

Task 2 dùng phép tính cosine vectorized, loại query khỏi kết quả và sắp xếp theo
score giảm dần, token tăng dần khi bằng điểm. Xem [Task 2](docs/TASK2.md).

Task 3 dùng feature Word2Vec đối xứng và Logistic Regression cân bằng lớp. Trên
1.173 cặp ViCon covered, accuracy = 0.7724 và macro-F1 = 0.7713; CSV vẫn giữ đủ
1.400 dòng. Xem [kết quả Task 3](docs/TASK3_RESULTS.md).

## Phân công

| Thành viên | Công việc | Thư mục |
|---|---|---|
| Member 1 | Cosine similarity, Pearson/Spearman, ViSim-400 | `src/task1/` |
| Member 2 | Vectorized top-k nearest words | `src/task2/` |
| Member 3 | Logistic Regression, Precision/Recall/F1, ViCon-400 | `src/task3/` |

Hướng dẫn chi tiết: [docs/TEAM_TASK_GUIDE.md](docs/TEAM_TASK_GUIDE.md).

## API dùng chung

```python
from src.common import get_coverage, load_embeddings, load_visim

embeddings = load_embeddings("data/word2vec/W2V_150.txt")
pairs = load_visim("data/ViSim-400/Visim-400.txt")

print(embeddings.size)       # 77021
print(embeddings.dimension)  # 150
print(embeddings.contains("sinh_viên"))
print(get_coverage(pairs, embeddings))
```

Các API chính:

- `load_embeddings(path)`
- `get_vector(word)`, `contains(word)`
- `get_vocabulary()`, `get_embedding_matrix()`
- `load_visim(path)`
- `load_vicon(path)`
- `load_training_pairs(synonym_path, antonym_path)`
- `get_coverage(pairs, embeddings)`

Từ OOV không được tự động thay bằng zero vector. Chi tiết xem
[docs/API_CONTRACT.md](docs/API_CONTRACT.md).

## Cấu trúc project

```text
WordSimilarity/
|-- data/                 # Word2Vec và các dataset gốc
|-- docs/                 # Hướng dẫn API, phân công và tích hợp
|-- results/              # Kết quả thực nghiệm
|-- src/
|   |-- common/           # Code dùng chung đã hoàn thành
|   |-- task1/            # Member 1
|   |-- task2/            # Member 2
|   `-- task3/            # Member 3
|-- tests/
|-- main.py
|-- pytest.ini
`-- requirements.txt
```

## Thông tin dữ liệu đã xác minh

| Dữ liệu | Thống kê |
|---|---|
| Word2Vec | 77.021 từ, 150 chiều |
| ViSim-400 | 400 cặp, 344 cặp có đủ embedding |
| ViCon-400 | 1.400 cặp, 1.173 cặp có đủ embedding |
| Training | 11.558 SYN, 2.000 ANT, 4 dòng không hợp lệ |

Loader chỉ báo cáo duplicate, nhãn mâu thuẫn và OOV; không tự ý xóa hay sửa dữ
liệu. ViCon có chồng lặp với training, vì vậy Task 3 phải báo cáo protocol rõ
ràng và không được dự đoán bằng label lookup.

## Tài liệu

- [Hướng dẫn từng thành viên](docs/TEAM_TASK_GUIDE.md)
- [Hợp đồng API chung](docs/API_CONTRACT.md)
- [Quy trình Git và tích hợp](docs/INTEGRATION_GUIDE.md)
- [Task 1: cosine và ViSim](docs/TASK1.md)
- [Task 2: k-nearest words](docs/TASK2.md)
- [Task 3: kết quả classifier](docs/TASK3_RESULTS.md)
