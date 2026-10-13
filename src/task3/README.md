# Task 3 — Synonym–Antonym Classification

Module này huấn luyện baseline Logistic Regression từ Word2Vec và đánh giá trên
ViCon-400. Dự đoán chỉ dựa trên feature embedding, không dùng label lookup.

## Chạy

Từ thư mục gốc repository:

```powershell
python main.py classify
```

Đổi vị trí output nếu cần:

```powershell
python main.py classify --output results/my_task3_predictions.csv
```

Một lần chạy tạo ba file cùng prefix:

- `*_predictions.csv`: đủ 1.400 dòng ViCon, kể cả OOV;
- `*_predictions_metrics.json`: protocol, coverage, diagnostics, overlap và metric;
- `*_predictions_report.md`: báo cáo ngắn có thể đọc trực tiếp.

## Feature và model

Với hai vector `v1`, `v2`, feature là:

```text
[abs(v1 - v2), v1 * v2, cosine(v1, v2)]
```

Feature có 301 chiều với embedding 150 chiều và không đổi khi đảo thứ tự hai
từ. Model là sklearn `Pipeline(StandardScaler, LogisticRegression)` với:

- `class_weight="balanced"`;
- `random_state=42`;
- `max_iter=1000`;
- scaler chỉ được fit trên training data.

## Protocol dữ liệu

- Bốn dòng training malformed được loader ghi nhận và không thể tạo pair.
- Duplicate và bảy cặp nhãn mâu thuẫn hợp lệ được **giữ nguyên và báo cáo**.
- Cặp OOV không được thay bằng zero vector; vẫn có trong CSV với `status=oov`
  nhưng không tham gia train hoặc metric.
- Overlap train/test được audit theo cặp không thứ tự, chỉ để báo cáo.
- Precision, Recall, F1, macro-F1 và confusion matrix được tính trên covered
  ViCon subset; thứ tự nhãn confusion matrix là `ANT`, `SYN`.

## API chính

```python
from src.task3 import (
    audit_overlap,
    build_feature_matrix,
    evaluate_predictions,
    pair_features,
    run_task3,
    train_classifier,
)
```

## Kiểm thử

```powershell
python -m pytest -q tests/task3
python -m pytest -q
```

Test bao phủ tính đối xứng/shape của feature, OOV, overlap đảo chiều, cấu hình
pipeline, tính tái lập, metric trên confusion matrix biết trước và workflow tạo
đủ artifact.

## Giới hạn diễn giải

ViCon chồng lặp rất lớn với training data khi bỏ qua thứ tự hai từ. Vì vậy kết
quả theo protocol bắt buộc của đề không phải đánh giá generalization sạch trên
các cặp chưa từng thấy. Báo cáo sinh ra luôn nêu rõ giới hạn này.

Kết quả chạy chuẩn hiện tại được ghi tại
[`docs/TASK3_RESULTS.md`](../../docs/TASK3_RESULTS.md).
