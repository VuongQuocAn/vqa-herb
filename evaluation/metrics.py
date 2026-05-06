"""evaluation/metrics.py — §11.2–11.3
evaluate(): chạy Beam Search, tính Val Loss, BLEU-4, ROUGE-L, VQA Accuracy.
compute_comprehensive_metrics(): tính thêm BLEU-1, BLEU-2, METEOR.
"""
import time
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

from configs.config import DEVICE
from training.trainer import _fmt_time


def normalize_answer(text: str) -> str:
    """Chuẩn hóa câu trả lời — chữ thường, bỏ dấu câu, xóa gạch dưới PhoBERT."""
    return text.lower().strip().rstrip(".,!?;:").replace("_", " ").strip()


def evaluate(
    model,
    loader,
    answer_vocab,
    criterion=None,
    beam_size: int = 1,
    device: torch.device = DEVICE,
) -> dict:
    """
    Đánh giá mô hình trên một tập dữ liệu.
    Trả về: bleu4, rouge_l, vqa_acc, val_loss (nếu có criterion).
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    actual_model.eval()

    rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    smooth = SmoothingFunction().method4

    all_refs   = []
    all_hyps   = []
    exact_hits = 0
    total      = 0
    total_loss = 0.0
    eval_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            pixel_values   = batch["image"].to(device, non_blocking=True)
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            answer_ids     = batch["answer_ids"].to(device, non_blocking=True)

            if criterion is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits  = model(pixel_values, input_ids, attention_mask, answer_ids)
                    targets = answer_ids[:, 1:]
                    loss    = criterion(logits, targets)
                total_loss += loss.item()

            answer_ids_cpu = answer_ids.cpu()
            for i in range(pixel_values.size(0)):
                pred_ids  = actual_model.predict(
                    pixel_values[i:i+1], input_ids[i:i+1], attention_mask[i:i+1],
                    beam_size=beam_size,
                )
                pred_text = answer_vocab.decode(pred_ids)
                ref_text  = answer_vocab.decode(answer_ids_cpu[i].tolist())
                pred_norm = normalize_answer(pred_text)
                ref_norm  = normalize_answer(ref_text)

                all_hyps.append(pred_norm.split())
                all_refs.append([ref_norm.split()])

                if pred_norm == ref_norm:
                    exact_hits += 1
                total += 1

            if batch_idx % 20 == 0 or batch_idx == len(loader):
                print(f"  Eval {batch_idx:>4}/{len(loader)} | {_fmt_time(time.time() - eval_start)}")

    bleu4 = corpus_bleu(all_refs, all_hyps,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)
    rouge_l_scores = [
        rouge.score(" ".join(ref[0]), " ".join(hyp))["rougeL"].fmeasure
        for ref, hyp in zip(all_refs, all_hyps)
    ]
    rouge_l  = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    vqa_acc  = exact_hits / total if total > 0 else 0.0
    val_loss = total_loss / len(loader) if criterion is not None else None

    print(f"\n  Kết quả ({total} mẫu | {_fmt_time(time.time() - eval_start)}):")
    if val_loss is not None:
        print(f"    Val Loss  : {val_loss:.4f}  <- Tiêu chí chọn checkpoint")
    print(f"    BLEU-4    : {bleu4:.4f}")
    print(f"    ROUGE-L   : {rouge_l:.4f}")
    print(f"    VQA Acc   : {vqa_acc:.4f}  ({exact_hits}/{total})")

    return {"val_loss": val_loss, "bleu4": bleu4, "rouge_l": rouge_l, "vqa_acc": vqa_acc}


def compute_comprehensive_metrics(model, loader, vocab, device: torch.device = DEVICE) -> dict:
    """Tính BLEU-1, BLEU-2, BLEU-4, ROUGE-L, METEOR trên tập dữ liệu."""
    model.eval()
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    all_refs_tokens, all_hyps_tokens = [], []
    all_refs_text,   all_hyps_text   = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Thu thập dự đoán"):
            pixel_values   = batch["image"].to(device, non_blocking=True)
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            answer_ids     = batch["answer_ids"]

            for i in range(pixel_values.size(0)):
                pred_ids  = actual_model.predict(
                    pixel_values[i:i+1], input_ids[i:i+1], attention_mask[i:i+1], beam_size=1
                )
                pred_text = vocab.decode(pred_ids)
                ref_text  = vocab.decode(answer_ids[i].tolist())
                pred_norm = normalize_answer(pred_text)
                ref_norm  = normalize_answer(ref_text)
                all_hyps_tokens.append(pred_norm.split())
                all_refs_tokens.append([ref_norm.split()])
                all_hyps_text.append(pred_norm)
                all_refs_text.append(ref_norm)

    smooth = SmoothingFunction().method1
    bleu1  = corpus_bleu(all_refs_tokens, all_hyps_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2  = corpus_bleu(all_refs_tokens, all_hyps_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu4  = corpus_bleu(all_refs_tokens, all_hyps_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    rouge_scores = [
        rouge.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(all_refs_text, all_hyps_text)
    ]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    meteor_scores = [
        meteor_score(ref_t, hyp_t)
        for ref_t, hyp_t in zip(all_refs_tokens, all_hyps_tokens)
    ]
    meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-4": bleu4, "ROUGE-L": rouge_l, "METEOR": meteor}
