# Integration and Git Guide

## Mục tiêu

Ba task phát triển song song trên shared Stage 0. Quy trình này giảm conflict,
giữ dataset bất biến và buộc integration test chạy trước merge.

> Thư mục hiện tại chưa được khởi tạo Git tại thời điểm Stage 0 được xây dựng.
> Người quản lý repository thực hiện `git init`/cấu hình remote nếu cần; Agent
> không tự tạo commit, push hoặc thay đổi remote.

## Ownership

| Phạm vi | Owner |
|---|---|
| `src/common/**`, contract chung | nhóm/maintainer |
| `src/task1/**`, `tests/task1/**` | Member 1 |
| `src/task2/**`, `tests/task2/**` | Member 2 |
| `src/task3/**`, `tests/task3/**` | Member 3 |
| `main.py`, dependency, README | integration owner |
| `data/**` | read-only đối với cả nhóm |

Stage 0 đã tạo `cli.py` placeholder trong từng task. Mỗi owner thay nội dung
`run(args)` của chính mình. Không cần chờ thuật toán của task khác.

## Branch và pull request

Sau khi repository Git/remote sẵn sàng:

```bash
git switch main
git pull --ff-only
git switch -c feature/taskN-short-name
```

Trong quá trình làm:

```bash
git status --short
python -m pytest -q
git add <chỉ-file-thuộc-task>
git commit -m "feat(taskN): concise description"
git push -u origin feature/taskN-short-name
```

Không commit Word2Vec, dataset đã sửa, virtual environment, cache hoặc output
tạm. Nếu output thực nghiệm cần nộp, chỉ commit file nhỏ đã thống nhất.

PR phải nêu:

- thuật toán và lựa chọn edge case;
- command tái tạo kết quả;
- test đã chạy và kết quả;
- coverage/OOV;
- file output;
- giới hạn hoặc protocol có thể làm metric thiên lệch.

## Khi cần thay shared API

1. Mô tả use case và thay đổi contract cho cả nhóm.
2. Tạo branch/commit riêng chỉ cho shared change.
3. Giữ backward compatibility nếu có thể.
4. Cập nhật `docs/API_CONTRACT.md` và test common.
5. Merge shared change trước; ba branch rebase/merge lại sau.

Không đặt task-specific cosine, feature engineering hoặc classifier vào common.

## Kết nối CLI

`main.py` parse ba command và gọi đúng một hàm:

```text
similarity -> src.task1.cli.run(args)
nearest    -> src.task2.cli.run(args)
classify   -> src.task3.cli.run(args)
```

Contract của `run`:

```python
def run(args: argparse.Namespace) -> int:
    # validate, load, execute, write/print
    return 0
```

Không gọi `sys.exit` bên trong task. Nếu bổ sung CLI option, owner đề xuất patch
nhỏ vào `main.py`; không đổi option hiện có. Các argument sẵn có:

- similarity: `embeddings`, `dataset`, `output`;
- nearest: `word`, `k`, `embeddings`;
- classify: `embeddings`, `synonyms`, `antonyms`, `test_dir`, `output`.

## Kiểm tra trước merge

Từ repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --help
python main.py similarity --help
python main.py nearest --help
python main.py classify --help
```

Sau khi task tương ứng hoàn tất, chạy command thật của task đó và kiểm tra:

- exit code 0;
- file output tồn tại và có đúng schema/count;
- không có `NaN` bất ngờ;
- OOV được đánh dấu, không biến thành zero vector;
- dataset gốc không thay đổi;
- command chạy lại cho kết quả giống nhau nếu có seed.

## Thứ tự tích hợp đề xuất

Các PR độc lập nên có thể merge theo bất kỳ thứ tự nào. Integration owner:

1. merge Stage 0;
2. review/merge từng task đã pass riêng;
3. sau mỗi merge, chạy toàn bộ test;
4. sau PR cuối, chạy cả ba CLI;
5. tổng hợp README/report từ output thật.

Nếu hai PR cùng sửa `main.py`, giữ nguyên argument contract hiện có và ghép chỉ
phần option bổ sung. Tuyệt đối không giải quyết conflict bằng cách lấy toàn bộ
phiên bản một bên rồi làm mất command của bên kia.

## Definition of integrated

- cả ba `run(args)` không còn placeholder;
- toàn bộ unit/integration test pass;
- ba command chạy độc lập từ clean environment;
- dependency được khai báo trong `requirements.txt`;
- output/report có thể tái tạo từ command đã ghi;
- không có thay đổi ngoài ý muốn trong `data/`;
- README phản ánh metric thực tế và các giới hạn leakage/OOV.
