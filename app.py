"""
app.py — Gradio Web UI Demo cho giảng viên kiểm tra tại local.

Sử dụng:
    python app.py
    python app.py --decoder lstm --port 7860
"""
import argparse
import os
import pickle

import gradio as gr
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

# ── Biến toàn cục cho model (tải 1 lần) ──────────────────────────────────────
_model        = None
_tokenizer    = None
_answer_vocab = None


def _load_model(decoder_type: str = "transformer"):
    global _model, _tokenizer, _answer_vocab

    with open(VOCAB_PATH, "rb") as f:
        _answer_vocab = pickle.load(f)

    _tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    _model = VQAModel(vocab_size=len(_answer_vocab), decoder_type=decoder_type).to(DEVICE)
    for phase in [2, 1]:
        tag = f"vqa_{decoder_type}_phase{phase}_best.pt"
        if os.path.exists(os.path.join(CHECKPOINT_DIR, tag)):
            load_checkpoint_if_exists(tag, _model)
            print(f"Đã nạp checkpoint: {tag}")
            break
    _model.eval()
    print(f"Model [{decoder_type.upper()}] sẵn sàng trên {DEVICE}.")


# ── Hàm dự đoán cho Gradio ───────────────────────────────────────────────────
def predict(image: Image.Image, question: str) -> str:
    """Hàm nhận ảnh PIL và câu hỏi, trả về câu trả lời dạng chuỗi."""
    if image is None:
        return "Vui lòng tải lên một ảnh dược liệu."
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi."

    pixel_values = val_test_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    enc = _tokenizer(
        question, max_length=MAX_QUESTION_LEN,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        pred_ids = _model.predict(pixel_values, input_ids, attention_mask, beam_size=3)

    return _answer_vocab.decode(pred_ids)


# ── Giao diện Gradio ──────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="VQA Dược Liệu Việt Nam", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # VQA Dược Liệu Việt Nam
            **Mô hình**: DINOv2 + PhoBERT + Co-Attention Fusion + Decoder
            **Tải lên ảnh dược liệu và đặt câu hỏi bằng tiếng Việt!**
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Ảnh Dược Liệu",
                    sources=["upload", "clipboard"],
                )
                question_input = gr.Textbox(
                    label="Câu hỏi",
                    placeholder="Ví dụ: Đây là cây gì? | Công dụng của loại này là gì?",
                    lines=2,
                )
                submit_btn = gr.Button("Trả lời", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label="Câu trả lời của mô hình",
                    lines=3,
                    interactive=False,
                )

        # Ví dụ mẫu
        # [TODO] Bổ sung 2-3 ảnh thật vào data/sample_images/ rồi cập nhật đường dẫn dưới đây
        gr.Examples(
            examples=[
                ["data/sample_images/placeholder.jpg", "Đây là cây gì?"],
                ["data/sample_images/placeholder.jpg", "Công dụng của loại dược liệu này là gì?"],
            ],
            inputs=[image_input, question_input],
            label="Ví dụ mẫu",
        )

        submit_btn.click(
            fn=predict,
            inputs=[image_input, question_input],
            outputs=answer_output,
        )
        question_input.submit(
            fn=predict,
            inputs=[image_input, question_input],
            outputs=answer_output,
        )

        gr.Markdown(
            """
            ---
            *Đồ án cuối kỳ môn Học Sâu (Deep Learning) — TDTU*
            """
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="VQA Herb Gradio App")
    parser.add_argument("--decoder", choices=["lstm", "transformer"], default="transformer")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true", help="Tạo public link qua Gradio")
    args = parser.parse_args()

    set_seed()
    _load_model(decoder_type=args.decoder)

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
