from dataclasses import dataclass
import numpy as np
from src.common import OutOfVocabularyError


@dataclass(frozen=True)
class Neighbor:
    word: str
    similarity: float


class NearestWordSearcher:
    """Xử lý tìm kiếm top-k từ gần nhất dựa trên Cosine Similarity chuẩn hóa (Vectorized)."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vocab = list(embeddings.get_vocabulary())
        self.vocab_array = np.asarray(self.vocab, dtype=np.str_)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Lấy ma trận embedding nguyên bản và chuẩn hóa L2 norm sẵn
        raw_matrix = embeddings.get_embedding_matrix().astype(np.float32)
        
        # Tính L2 norm cho từng dòng, tránh chia cho 0
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        self.zero_norm_mask = norms[:, 0] == 0
        norms[self.zero_norm_mask] = 1.0  # Tránh lỗi NaN khi row norm = 0
        
        # Cache ma trận đã chuẩn hóa
        self.norm_matrix = raw_matrix / norms

    def find_nearest(self, word: str, k: int) -> list[Neighbor]:
        # 1. Validate k
        if k <= 0:
            raise ValueError("Tham số k phải là số nguyên dương lớn hơn 0.")

        # 2. Kiểm tra từ OOV
        if not self.embeddings.contains(word):
            raise OutOfVocabularyError(f"Từ '{word}' không có trong từ điển (OOV).")

        query_idx = self.word_to_idx[word]
        if self.zero_norm_mask[query_idx]:
            raise ValueError(
                f"Không thể tính cosine cho từ '{word}' vì vector có norm bằng 0."
            )
        q_vec = self.norm_matrix[query_idx]

        # 3. Tính Cosine Similarity hàng loạt: s = E_norm @ q
        scores = self.norm_matrix @ q_vec

        # 4. Loại từ truy vấn và các vector zero-norm vì cosine không xác định.
        scores[self.zero_norm_mask] = -np.inf
        scores[query_idx] = -np.inf

        # 5. Xử lý k quá lớn (clamp theo số candidate có cosine hợp lệ).
        max_possible_k = int(np.isfinite(scores).sum())
        effective_k = min(k, max_possible_k)

        if effective_k <= 0:
            return []

        # 6. Sắp xếp toàn cục theo score giảm dần rồi token tăng dần.
        # np.lexsort bảo đảm tie-break đúng cả tại ranh giới top-k; argpartition
        # đơn thuần có thể chọn ngẫu nhiên một phần tử trong nhóm đồng hạng.
        ordered_indices = np.lexsort((self.vocab_array, -scores))
        top_indices = [
            int(idx)
            for idx in ordered_indices
            if np.isfinite(scores[idx])
        ][:effective_k]

        # 7. Trả về kết quả
        return [
            Neighbor(word=self.vocab[idx], similarity=float(scores[idx]))
            for idx in top_indices
        ]


def nearest_words(embeddings, word: str, k: int) -> list[Neighbor]:
    """Hàm tiện ích bọc ngoài class Searcher."""
    searcher = NearestWordSearcher(embeddings)
    return searcher.find_nearest(word, k)
