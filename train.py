"""
train.py — Entry point huấn luyện mô hình VQA Dược liệu.

Sử dụng:
    python train.py --decoder transformer --phase 1
    python train.py --decoder lstm --phase 2
"""
import argparse
import os
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from configs.config import (
    HF_DATASET_ID, HF_HUB_TOKEN, CHECKPOINT_DIR,
    PHOBERT_MODEL_NAME, FIELD_ANSWER, FIELD_FILE_NAME,
    VOCAB_PATH, DEVICE, EPOCHS_PHASE1, EPOCHS_PHASE2,
    GRAD_ACCUMULATION, EARLY_STOPPING_PATIENCE, set_seed,
)
from data.vocab   import AnswerVocab
from data.dataset import prefetch_images, filter_valid_samples, build_dataloaders
from models.vqa_model  import VQAModel
from training.loss      import LabelSmoothedCrossEntropy
from training.optimizer import build_optimizer_phase1, build_optimizer_phase2
from training.trainer   import train_one_epoch, _fmt_time
from evaluation.metrics import evaluate
from utils.checkpoint   import save_checkpoint, load_checkpoint_if_exists, plot_loss_curves

import time


def main():
    parser = argparse.ArgumentParser(description="VQA Herb Training")
    parser.add_argument("--decoder", choices=["lstm", "transformer"], default="transformer")
    parser.add_argument("--phase",   type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    set_seed()
    print(f"Device: {DEVICE}")
    print(f"Decoder: {args.decoder} | Phase: {args.phase}")

    # ── Tải dữ liệu ──────────────────────────────────────────────────────────
    import huggingface_hub
    if HF_HUB_TOKEN:
        huggingface_hub.login(token=HF_HUB_TOKEN)
    raw_dataset = load_dataset(HF_DATASET_ID)

    train_data = raw_dataset["train"]
    val_data   = raw_dataset["validation"]
    test_data  = raw_dataset["test"]

    # Prefetch ảnh về local
    prefetch_images([train_data, val_data, test_data])

    # ── Vocab & Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "rb") as f:
            answer_vocab = pickle.load(f)
        print(f"Vocab đã tải từ {VOCAB_PATH} ({len(answer_vocab):,} từ)")
    else:
        answer_vocab = AnswerVocab()
        answer_vocab.build([s[FIELD_ANSWER] for s in train_data])
        with open(VOCAB_PATH, "wb") as f:
            pickle.dump(answer_vocab, f)
        print(f"Vocab đã xây và lưu tại {VOCAB_PATH} ({len(answer_vocab):,} từ)")

    vocab_size = len(answer_vocab)

    # ── DataLoader ───────────────────────────────────────────────────────────
    train_samples = filter_valid_samples(train_data)
    val_samples   = filter_valid_samples(val_data)
    test_samples  = filter_valid_samples(test_data)
    train_loader, val_loader, _, _, _, _ = build_dataloaders(
        train_samples, val_samples, test_samples, tokenizer, answer_vocab
    )

    # ── Model & Loss ─────────────────────────────────────────────────────────
    model     = VQAModel(vocab_size=vocab_size, decoder_type=args.decoder).to(DEVICE)
    criterion = LabelSmoothedCrossEntropy(vocab_size=vocab_size).to(DEVICE)
    scaler    = torch.amp.GradScaler("cuda")

    ckpt_tag  = f"vqa_{args.decoder}_phase{args.phase}_best.pt"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_tag)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if args.phase == 1:
        model.freeze_encoders()
        total_steps = (len(train_loader) // GRAD_ACCUMULATION) * EPOCHS_PHASE1
        optimizer, scheduler = build_optimizer_phase1(model, total_steps)
        start_epoch, train_losses, val_losses, best_val_loss, patience = \
            load_checkpoint_if_exists(ckpt_tag, model, optimizer, scheduler)

        print(f"\n{'='*60}\nGIAI ĐOẠN 1 — {args.decoder.upper()} — FREEZE ENCODER\n{'='*60}")
        for epoch in range(start_epoch, EPOCHS_PHASE1 + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                         criterion, scaler, epoch)
            train_losses.append(train_loss)
            val_metrics = evaluate(model, val_loader, answer_vocab,
                                   criterion=criterion, beam_size=1)
            val_losses.append(val_metrics["val_loss"])

            print(f"\nEpoch {epoch} | Train: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"BLEU-4: {val_metrics['bleu4']:.4f} | "
                  f"ETA: {_fmt_time(time.time()-t0)}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience = 0
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                                train_losses, val_losses, patience, ckpt_path)
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"Early Stopping — dừng sau {patience} epoch không cải thiện.")
                    break

        plot_loss_curves(train_losses, val_losses, phase=1)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    else:
        model.unfreeze_encoders_partial(n_img_blocks=2, n_txt_layers=2)
        total_steps = (len(train_loader) // GRAD_ACCUMULATION) * EPOCHS_PHASE2
        optimizer, scheduler = build_optimizer_phase2(model, total_steps)

        # Load Phase 1 checkpoint nếu Phase 2 chưa có
        p1_tag = f"vqa_{args.decoder}_phase1_best.pt"
        start_epoch, train_losses, val_losses, best_val_loss, patience = \
            load_checkpoint_if_exists(ckpt_tag, model, optimizer, scheduler)
        if start_epoch == 1:
            load_checkpoint_if_exists(p1_tag, model)
            best_val_loss = float("inf")

        print(f"\n{'='*60}\nGIAI ĐOẠN 2 — {args.decoder.upper()} — UNFREEZE ENCODER\n{'='*60}")
        for epoch in range(start_epoch, EPOCHS_PHASE2 + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                         criterion, scaler, epoch)
            train_losses.append(train_loss)
            val_metrics = evaluate(model, val_loader, answer_vocab,
                                   criterion=criterion, beam_size=1)
            val_losses.append(val_metrics["val_loss"])

            print(f"\nEpoch {epoch} | Train: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"BLEU-4: {val_metrics['bleu4']:.4f} | "
                  f"ETA: {_fmt_time(time.time()-t0)}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience = 0
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                                train_losses, val_losses, patience, ckpt_path)
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"Early Stopping — dừng sau {patience} epoch không cải thiện.")
                    break

        plot_loss_curves(train_losses, val_losses, phase=2)


if __name__ == "__main__":
    main()
