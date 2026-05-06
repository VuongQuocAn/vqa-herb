"""
evaluate.py — Entry point chạy đánh giá đầy đủ trên tập Test.

Sử dụng:
    python evaluate.py --decoder transformer
    python evaluate.py --decoder lstm
"""
import argparse
import os
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from configs.config import (
    HF_DATASET_ID, HF_HUB_TOKEN, CHECKPOINT_DIR,
    PHOBERT_MODEL_NAME, FIELD_ANSWER, VOCAB_PATH,
    DEVICE, set_seed,
)
from data.vocab   import AnswerVocab
from data.dataset import filter_valid_samples, build_dataloaders
from models.vqa_model import VQAModel
from evaluation.metrics import evaluate, compute_comprehensive_metrics
from utils.checkpoint   import load_checkpoint_if_exists


def main():
    parser = argparse.ArgumentParser(description="VQA Herb Evaluation")
    parser.add_argument("--decoder", choices=["lstm", "transformer"], default="transformer")
    args = parser.parse_args()

    set_seed()
    print(f"Device: {DEVICE} | Decoder: {args.decoder}")

    # ── Tải dữ liệu ──────────────────────────────────────────────────────────
    import huggingface_hub
    if HF_HUB_TOKEN:
        huggingface_hub.login(token=HF_HUB_TOKEN)
    raw_dataset = load_dataset(HF_DATASET_ID)

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    with open(VOCAB_PATH, "rb") as f:
        answer_vocab = pickle.load(f)
    print(f"Vocab: {len(answer_vocab):,} từ")

    val_samples  = filter_valid_samples(raw_dataset["validation"])
    test_samples = filter_valid_samples(raw_dataset["test"])
    train_samples = filter_valid_samples(raw_dataset["train"])
    _, val_loader, test_loader, _, _, _ = build_dataloaders(
        train_samples, val_samples, test_samples, tokenizer, answer_vocab
    )

    # ── Nạp checkpoint tốt nhất ───────────────────────────────────────────────
    model = VQAModel(vocab_size=len(answer_vocab), decoder_type=args.decoder).to(DEVICE)

    # Ưu tiên Phase 2, fallback về Phase 1
    for phase in [2, 1]:
        tag = f"vqa_{args.decoder}_phase{phase}_best.pt"
        if os.path.exists(os.path.join(CHECKPOINT_DIR, tag)):
            load_checkpoint_if_exists(tag, model)
            print(f"Đã nạp checkpoint: {tag}")
            break
    model.eval()

    # ── Đánh giá ─────────────────────────────────────────────────────────────
    print("\n--- BLEU / ROUGE-L / VQA Acc ---")
    metrics = compute_comprehensive_metrics(model, test_loader, answer_vocab, DEVICE)

    print(f"\n{'='*50}")
    print("KẾT QUẢ TRÊN TẬP TEST")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")


if __name__ == "__main__":
    main()
