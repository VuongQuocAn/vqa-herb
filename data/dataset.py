"""data/dataset.py — §3.3–3.5
HerbVQADataset, image transforms, DataLoader helpers.
Hỗ trợ tải ảnh cache từ HuggingFace Hub khi không có sẵn ở local.
"""
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from configs.config import (
    HF_DATASET_ID, HF_HUB_TOKEN,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MAX_QUESTION_LEN, MAX_ANSWER_LEN,
    FIELD_FILE_NAME, FIELD_QUESTION, FIELD_ANSWER,
    EOS_ID, PAD_ID,
    BATCH_SIZE, NUM_WORKERS,
)

# ── Đường dẫn cache ảnh ──────────────────────────────────────────────────────
IMAGE_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "image_cache"
)
HF_RAW_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main"
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Helpers tải ảnh ──────────────────────────────────────────────────────────
def _get_local_path(file_name: str) -> str | None:
    """Tìm đường dẫn local của ảnh, trả về None nếu chưa có."""
    for base in [IMAGE_CACHE_DIR]:
        p = os.path.join(base, file_name)
        if os.path.exists(p):
            return p
    return None


def _download_one(file_name: str) -> str | None:
    """Tải một ảnh từ HuggingFace Hub về cache."""
    cache_path = os.path.join(IMAGE_CACHE_DIR, file_name)
    if os.path.exists(cache_path):
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    headers = {"Authorization": f"Bearer {HF_HUB_TOKEN}"} if HF_HUB_TOKEN else {}
    try:
        resp = requests.get(f"{HF_RAW_URL}/{file_name}", headers=headers, timeout=30)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(resp.content)
        return cache_path
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def prefetch_images(splits: list, max_workers: int = 8) -> None:
    """Tải song song toàn bộ ảnh chưa có local trước khi huấn luyện."""
    missing = []
    for split in splits:
        for fname in split[FIELD_FILE_NAME]:
            if _get_local_path(fname) is None:
                missing.append(fname)
    missing = list(set(missing))

    if not missing:
        print("Tất cả ảnh đã có ở local.")
        return

    print(f"Đang tải {len(missing):,} ảnh còn thiếu ({max_workers} luồng)...")
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download_one, fn): fn for fn in missing}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tải ảnh"):
            try:
                fut.result()
            except Exception:
                failed.append(futures[fut])

    if failed:
        print(f"Cảnh báo: {len(failed)} ảnh tải thất bại.")
    else:
        print("Đã tải xong toàn bộ ảnh.")


def filter_valid_samples(hf_split) -> list[dict]:
    """Chỉ giữ lại các mẫu có ảnh tồn tại ở local."""
    return [s for s in hf_split if _get_local_path(s[FIELD_FILE_NAME]) is not None]


# ── Dataset ───────────────────────────────────────────────────────────────────
class HerbVQADataset(Dataset):
    """
    Dataset VQA Dược liệu.
    - Câu hỏi  : PhoBERT tokenizer (encoder nhận embedding 768 chiều).
    - Câu trả lời: AnswerVocab tùy chỉnh (decoder nhận không gian nhỏ ~5K từ).
    """

    def __init__(self, hf_split, transform, tokenizer, answer_vocab):
        self.data         = hf_split
        self.transform    = transform
        self.tokenizer    = tokenizer
        self.answer_vocab = answer_vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        local  = _get_local_path(sample[FIELD_FILE_NAME])
        image  = self.transform(Image.open(local).convert("RGB"))

        enc_q = self.tokenizer(
            sample[FIELD_QUESTION],
            max_length=MAX_QUESTION_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc_q["input_ids"].squeeze(0)
        attention_mask = enc_q["attention_mask"].squeeze(0)

        answer_ids = torch.tensor(
            self.answer_vocab.encode(sample[FIELD_ANSWER], max_len=MAX_ANSWER_LEN),
            dtype=torch.long,
        )

        eos_positions = (answer_ids == EOS_ID).nonzero(as_tuple=True)[0]
        answer_length = eos_positions[0].item() + 1 if len(eos_positions) > 0 else MAX_ANSWER_LEN

        return {
            "image":          image,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "answer_ids":     answer_ids,
            "answer_length":  torch.tensor(answer_length, dtype=torch.long),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────
def build_dataloaders(train_samples, val_samples, test_samples,
                      tokenizer, answer_vocab):
    """Tạo DataLoader cho 3 tập train/val/test."""
    train_ds = HerbVQADataset(train_samples, train_transform,    tokenizer, answer_vocab)
    val_ds   = HerbVQADataset(val_samples,   val_test_transform, tokenizer, answer_vocab)
    test_ds  = HerbVQADataset(test_samples,  val_test_transform, tokenizer, answer_vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
