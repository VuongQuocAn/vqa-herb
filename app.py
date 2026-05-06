"""
app.py — Gradio Web UI Demo cho giảng viên kiểm tra tại local.

Sử dụng:
    python app.py
    python app.py --decoder lstm --port 7860
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Thư mục gốc của app.py (luôn đúng bất kể chạy từ đâu)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SAMPLE_DIR = os.path.join(_APP_DIR, "data", "sample_images")

import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer

from configs.config import (
    CHECKPOINT_DIR, VOCAB_PATH, PHOBERT_MODEL_NAME,
    MAX_QUESTION_LEN, DEVICE, HF_HUB_REPO_ID, set_seed,
)
from data.dataset import val_test_transform
from models.vqa_model import VQAModel

# ── Biến toàn cục cho model (tải 1 lần) ──────────────────────────────────────
_model        = None
_tokenizer    = None
_answer_vocab = None


def _ensure_checkpoint(tag: str) -> str:
    """Đảm bảo checkpoint tồn tại ở local, tự tải từ HuggingFace nếu chưa có."""
    local_path = os.path.join(CHECKPOINT_DIR, tag)
    if not os.path.exists(local_path):
        from huggingface_hub import hf_hub_download
        print(f"Không tìm thấy {tag} ở local, đang tải từ HuggingFace...")
        hf_hub_download(
            repo_id=HF_HUB_REPO_ID,
            filename=tag,
            local_dir=CHECKPOINT_DIR,
        )
        print(f"Tải xong: {tag}")
    return local_path


def _load_model(decoder_type: str = "transformer"):
    global _model, _tokenizer, _answer_vocab

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'AnswerVocab':
                from data.vocab import AnswerVocab
                return AnswerVocab
            return super().find_class(module, name)

    with open(VOCAB_PATH, "rb") as f:
        _answer_vocab = CustomUnpickler(f).load()

    _tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    _model = VQAModel(vocab_size=len(_answer_vocab), decoder_type=decoder_type).to(DEVICE)
    model_prefix = "a1" if decoder_type == "lstm" else "a2"

    checkpoint_loaded = False
    for phase in [2, 1]:
        tag = f"vqa_{model_prefix}_phase{phase}_best.pt"
        try:
            local_path = _ensure_checkpoint(tag)
            ckpt = torch.load(local_path, map_location=DEVICE)
            _model.load_state_dict(ckpt["model"])
            print(f"Đã nạp checkpoint: {tag}")
            checkpoint_loaded = True
            break
        except Exception as e:
            print(f"Không thể tải {tag}: {e}")
            continue

    if not checkpoint_loaded:
        print("CẢNH BÁO: Không nạp được checkpoint nào! Model đang dùng trọng số ngẫu nhiên.")

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
        pred_ids, attn_map_2d = _model.predict_with_attention(
            pixel_values, input_ids, attention_mask, beam_size=3
        )

    answer_text = _answer_vocab.decode(pred_ids)
    attn_img    = _make_attention_overlay(image, attn_map_2d)
    return answer_text, attn_img


def _make_attention_overlay(image: Image.Image, attn_map: "torch.Tensor") -> Image.Image:
    """Vẽ heatmap attention (jet colormap) đè lên ảnh gốc, trả về PIL Image."""
    w, h = image.size
    attn_np = attn_map.numpy().astype(np.float32)  # (grid, grid)

    # Up-sample heatmap về đúng kích thước ảnh
    from PIL import Image as PILImage
    heatmap_pil = PILImage.fromarray(
        np.uint8(attn_np * 255)
    ).resize((w, h), PILImage.BILINEAR)
    heatmap_np  = np.array(heatmap_pil) / 255.0  # [0,1]

    # Áp colormap Jet
    colormap    = matplotlib.colormaps["jet"]
    heatmap_rgb = colormap(heatmap_np)[:, :, :3]   # (H,W,3) float [0,1]

    # Hòa trộn: 50% ảnh gốc + 50% heatmap
    orig_np     = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    blended     = 0.5 * orig_np + 0.5 * heatmap_rgb
    blended     = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return PILImage.fromarray(blended)


# ── Giao diện Gradio ─────────────────────────────────────────────────────────────────────────────
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
                gr.Markdown("#### 🔍 Attention Map — Mô hình đang nhìn vào đâu?")
                attn_output = gr.Image(
                    show_label=False,
                    type="pil",
                    interactive=False,
                )

        # Ví dụ mẫu
        gr.Examples(
            examples=[
                [os.path.join(_SAMPLE_DIR, "demo_10_P000001916.jpg"), "Đây là cây gì?"],
                [os.path.join(_SAMPLE_DIR, "demo_12_P000000048.jpg"), "Công dụng của loại dược liệu này là gì?"],
            ],
            inputs=[image_input, question_input],
            label="Ví dụ mẫu",
        )

        submit_btn.click(
            fn=predict,
            inputs=[image_input, question_input],
            outputs=[answer_output, attn_output],
        )
        question_input.submit(
            fn=predict,
            inputs=[image_input, question_input],
            outputs=[answer_output, attn_output],
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
