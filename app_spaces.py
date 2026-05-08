"""
app_spaces.py — Entry-point cho Hugging Face Spaces.

Khác với app.py (local), file này:
  - KHÔNG dùng argparse (HF Spaces không hỗ trợ CLI args)
  - Hard-code decoder_type = "transformer"
  - Đọc HF_TOKEN từ Spaces Secret (biến môi trường)
  - Gọi demo.launch() không có server_port / share
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
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

# ── Decoder mặc định cho Spaces ──────────────────────────────────────────────
DECODER_TYPE = os.getenv("DECODER_TYPE", "transformer")   # override bằng Spaces secret nếu cần

_model        = None
_tokenizer    = None
_answer_vocab = None


def _ensure_checkpoint(tag: str) -> str:
    """Tải checkpoint từ HuggingFace Hub nếu chưa có ở local (Spaces cache)."""
    local_path = os.path.join(CHECKPOINT_DIR, tag)
    if not os.path.exists(local_path):
        from huggingface_hub import hf_hub_download
        print(f"[Spaces] Đang tải {tag} từ HuggingFace Hub...")
        hf_hub_download(
            repo_id=HF_HUB_REPO_ID,
            filename=tag,
            local_dir=CHECKPOINT_DIR,
            token=os.getenv("HF_TOKEN"),   # đọc từ Spaces Secret
        )
        print(f"[Spaces] Tải xong: {tag}")
    return local_path


def _load_model():
    global _model, _tokenizer, _answer_vocab

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "AnswerVocab":
                from data.vocab import AnswerVocab
                return AnswerVocab
            return super().find_class(module, name)

    with open(VOCAB_PATH, "rb") as f:
        _answer_vocab = CustomUnpickler(f).load()

    _tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    _model = VQAModel(vocab_size=len(_answer_vocab), decoder_type=DECODER_TYPE).to(DEVICE)
    model_prefix = "a1" if DECODER_TYPE == "lstm" else "a2"

    for phase in [2, 1]:
        tag = f"vqa_{model_prefix}_phase{phase}_best.pt"
        try:
            local_path = _ensure_checkpoint(tag)
            ckpt = torch.load(local_path, map_location=DEVICE)
            _model.load_state_dict(ckpt["model"])
            print(f"[Spaces] Đã nạp checkpoint: {tag}")
            break
        except Exception as e:
            print(f"[Spaces] Không thể tải {tag}: {e}")

    _model.eval()
    print(f"[Spaces] Model [{DECODER_TYPE.upper()}] sẵn sàng trên {DEVICE}.")


def _make_attention_overlay(image: Image.Image, attn_map: "torch.Tensor") -> Image.Image:
    from PIL import Image as PILImage
    w, h = image.size
    attn_np = attn_map.numpy().astype(np.float32)
    heatmap_pil = PILImage.fromarray(np.uint8(attn_np * 255)).resize((w, h), PILImage.BILINEAR)
    heatmap_np  = np.array(heatmap_pil) / 255.0
    colormap    = matplotlib.colormaps["jet"]
    heatmap_rgb = colormap(heatmap_np)[:, :, :3]
    orig_np     = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    blended     = np.clip((0.5 * orig_np + 0.5 * heatmap_rgb) * 255, 0, 255).astype(np.uint8)
    return PILImage.fromarray(blended)


def predict(image: Image.Image, question: str):
    if image is None:
        return "Vui lòng tải lên một ảnh dược liệu.", None
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", None

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


def build_ui():
    sample_examples = []
    for fname, q in [
        ("demo_10_P000001916.jpg", "Đây là cây gì?"),
        ("demo_12_P000000048.jpg", "Công dụng của loại dược liệu này là gì?"),
    ]:
        p = os.path.join(_SAMPLE_DIR, fname)
        if os.path.exists(p):
            sample_examples.append([p, q])

    with gr.Blocks(title="VQA Dược Liệu Việt Nam", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌿 VQA Dược Liệu Việt Nam
            **Mô hình**: DINOv2 + PhoBERT + Co-Attention Fusion + Transformer Decoder

            Tải lên ảnh dược liệu và đặt câu hỏi bằng **tiếng Việt**!
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input   = gr.Image(type="pil", label="Ảnh Dược Liệu", sources=["upload", "clipboard"])
                question_input = gr.Textbox(
                    label="Câu hỏi",
                    placeholder="Ví dụ: Đây là cây gì? | Công dụng của loại này là gì?",
                    lines=2,
                )
                submit_btn = gr.Button("Trả lời", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="Câu trả lời của mô hình", lines=3, interactive=False)
                gr.Markdown("#### 🔍 Attention Map — Mô hình đang nhìn vào đâu?")
                attn_output = gr.Image(show_label=False, type="pil", interactive=False)

        if sample_examples:
            gr.Examples(examples=sample_examples, inputs=[image_input, question_input], label="Ví dụ mẫu")

        submit_btn.click(fn=predict, inputs=[image_input, question_input], outputs=[answer_output, attn_output])
        question_input.submit(fn=predict, inputs=[image_input, question_input], outputs=[answer_output, attn_output])

        gr.Markdown("---\n*Đồ án cuối kỳ môn Học Sâu (Deep Learning) — TDTU*")
    return demo


# ── Khởi động ────────────────────────────────────────────────────────────────
set_seed()
_load_model()
demo = build_ui()

# HF Spaces tự gọi demo.launch() — KHÔNG cần gọi thủ công
# Nhưng để chạy được nếu test bằng `python app_spaces.py`:
if __name__ == "__main__":
    demo.launch()
