"""data/vocab.py — §3.2
Class AnswerVocab: xây dựng bộ từ vựng riêng từ tập câu trả lời train.
"""
from collections import Counter


class AnswerVocab:
    """Bộ từ vựng tự xây từ câu trả lời tập Train."""

    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def build(self, answers: list[str], min_freq: int = 1) -> None:
        """Xây dựng vocab từ danh sách câu trả lời."""
        counter = Counter()
        for ans in answers:
            counter.update(ans.lower().strip().split())

        self.word2idx = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text: str, max_len: int) -> list[int]:
        """Mã hóa câu text thành list ID, có SOS/EOS/PAD."""
        tokens = [
            self.word2idx.get(w, self.word2idx["<UNK>"])
            for w in text.lower().strip().split()
        ]
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
        tokens = tokens[:max_len]
        tokens += [self.word2idx["<PAD>"]] * (max_len - len(tokens))
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Giải mã list ID về chuỗi text, bỏ SOS/EOS/PAD."""
        stop = {
            self.word2idx["<PAD>"],
            self.word2idx["<EOS>"],
            self.word2idx["<SOS>"],
        }
        return " ".join(
            self.idx2word.get(i, "<UNK>") for i in ids if i not in stop
        )

    def __len__(self) -> int:
        return len(self.word2idx)
