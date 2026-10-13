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
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Lấy ma trận embedding nguyên bản và chuẩn hóa L2 norm sẵn
        raw_matrix = embeddings.get_embedding_matrix().astype(np.float32)
        
        # Tính L2 norm cho từng dòng, tránh chia cho 0
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Tránh lỗi NaN khi row norm = 0
        
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
        q_vec = self.norm_matrix[query_idx]

        # 3. Tính Cosine Similarity hàng loạt: s = E_norm @ q
        scores = self.norm_matrix @ q_vec

        # 4. Gán score của từ truy vấn = -inf để không xuất hiện trong top-k
        scores[query_idx] = -np.inf

        # 5. Xử lý k quá lớn (clamp k tối đa bằng vocab_size - 1)
        max_possible_k = len(self.vocab) - 1
        effective_k = min(k, max_possible_k)

        if effective_k <= 0:
            return []

        # 6. Dùng np.argpartition để lấy candidate top-k nhanh nhất
        partition_indices = np.argpartition(scores, -effective_k)[-effective_k:]

        # 7. Sắp xếp lại top-k candidates chính xác:
        # Tie-break rule: Similarity giảm dần, nếu bằng nhau thì Token tăng dần (alphabet)
        candidates = [
            (idx, float(scores[idx]), self.vocab[idx])
            for idx, score in zip(partition_indices, scores[partition_indices])
            if np.isfinite(score)
        ]

        candidates.sort(key=lambda x: (-x[1], x[2]))

        # 8. Trả về kết quả
        return [
            Neighbor(word=token, similarity=sim)
            for _, sim, token in candidates[:effective_k]
        ]


def nearest_words(embeddings, word: str, k: int) -> list[Neighbor]:
    """Hàm tiện ích bọc ngoài class Searcher."""
    searcher = NearestWordSearcher(embeddings)
    return searcher.find_nearest(word, k)