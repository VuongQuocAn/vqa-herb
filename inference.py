"""
inference.py — Entry point dự đoán nhanh cho 1 ảnh + 1 câu hỏi.

Sử dụng:
    python inference.py --image data/sample_images/cam_thao.jpg --question "Đây là cây gì?"
    python inference.py --image path/to/img.jpg --question "Công dụng của cây này là gì?" --decoder transformer
"""
import argparse
import os
import pickle

import torch
from PIL import Image
from transformers import AutoTokenizer

from configs.config import (
    CHECKPOINT_DIR, VOCAB_PATH, PHOBERT_MODEL_NAME,
    MAX_QUESTION_LEN, DEVICE, set_seed,
)
from data.dataset import val_test_transform
from models.vqa_model import VQAModel
from utils.checkpoint import load_checkpoint_if_exists


def predict_single(image_path: str, question: str, model: VQAModel,
                   tokenizer, answer_vocab, device=DEVICE) -> str:
    """Dự đoán câu trả lời cho một ảnh và câu hỏi."""
    # Tiền xử lý ảnh
    image = Image.open(image_path).convert("RGB")
    pixel_values = val_test_transform(image).unsqueeze(0).to(device)

    # Tokenize câu hỏi
    enc = tokenizer(
        question, max_length=MAX_QUESTION_LEN,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Dự đoán
    pred_ids = model.predict(pixel_values, input_ids, attention_mask, beam_size=3)
    return answer_vocab.decode(pred_ids)


def main():
    parser = argparse.ArgumentParser(description="VQA Herb Inference")
    parser.add_argument("--image",    required=True, help="Đường dẫn tới file ảnh")
    parser.add_argument("--question", required=True, help="Câu hỏi về ảnh")
    parser.add_argument("--decoder",  choices=["lstm", "transformer"], default="transformer")
    args = parser.parse_args()

    set_seed()

    # Load vocab
    with open(VOCAB_PATH, "rb") as f:
        answer_vocab = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    # Load model
    model = VQAModel(vocab_size=len(answer_vocab), decoder_type=args.decoder).to(DEVICE)
    for phase in [2, 1]:
        tag = f"vqa_{args.decoder}_phase{phase}_best.pt"
        if os.path.exists(os.path.join(CHECKPOINT_DIR, tag)):
            load_checkpoint_if_exists(tag, model)
            break
    model.eval()

    # Dự đoán
    answer = predict_single(args.image, args.question, model, tokenizer, answer_vocab)

    print(f"\n{'='*50}")
    print(f"Ảnh    : {args.image}")
    print(f"Câu hỏi: {args.question}")
    print(f"Trả lời: {answer}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
