# 🌿 VQA Dược Liệu Việt Nam

> **Mô hình Visual Question Answering (VQA) cho dược liệu Việt Nam**  
> Kiến trúc module: **DINOv2** + **PhoBERT** + **Co-Attention Fusion** + **LSTM / Transformer Decoder**

---

<!-- [TODO] Chèn ảnh banner/demo vào đây sau khi có ảnh thật
![Demo Screenshot](docs/images/demo_screenshot.png)
-->

## 📋 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cài đặt](#-cài-đặt)
- [Sử dụng nhanh](#-sử-dụng-nhanh)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Demo Web UI](#-demo-web-ui)
- [Dataset & Checkpoint](#-dataset--checkpoint)

---

## 🎯 Giới thiệu

Dự án xây dựng hệ thống trả lời câu hỏi dựa trên hình ảnh (VQA) cho bộ dữ liệu dược liệu y học cổ truyền Việt Nam. Mô hình có khả năng nhận dạng loài cây, mô tả đặc điểm hình thái và trả lời các câu hỏi về công dụng y học.

**Môn học:** Học Sâu (Deep Learning)  
**Dữ liệu:** [tracuuduoclieu.vn](https://tracuuduoclieu.vn)  
**Môi trường huấn luyện:** Kaggle (GPU T4)

---

## 🏗 Kiến trúc mô hình

```
Ảnh (224×224) ──► ImageEncoder (DINOv2-base) ──► patch_embeddings (256×768)
                                                          │
                                                ┌─────────▼──────────┐
Câu hỏi (VI) ──► TextEncoder (PhoBERT-base-v2) │  Co-Attention      │──► fusion_memory
                ──► token_embeddings (64×768) ──┤  Fusion Module     │──► fusion_cls
                                                └────────────────────┘
                                                          │
                                           ┌──────────────┴─────────────┐
                                           │                             │
                                    LSTMDecoder (A1)       TransformerDecoder (A2)
                                    + BahdanauAttention     + PositionalEncoding
                                           │                             │
                                           └──────────────┬─────────────┘
                                                          ▼
                                               Câu trả lời (Tiếng Việt)
```

<!-- [TODO] Thêm ảnh sơ đồ kiến trúc chính thức (từ báo cáo hoặc draw.io) vào docs/images/architecture.png
![Architecture](docs/images/architecture.png)
-->

---

## ⚙️ Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/<YOUR_USERNAME>/vqa-herb.git
cd vqa-herb
```

### 2. Tạo môi trường ảo và cài thư viện
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
```bash
cp .env.example .env
# Mở .env và điền HF_TOKEN của bạn
```

---

## 🚀 Sử dụng nhanh

### Huấn luyện
```bash
# Giai đoạn 1 — Freeze Encoder, chỉ train Fusion + Decoder
python train.py --decoder transformer --phase 1

# Giai đoạn 2 — Unfreeze một phần Encoder, fine-tune toàn bộ
python train.py --decoder transformer --phase 2

# Tương tự cho LSTM Decoder
python train.py --decoder lstm --phase 1
```

### Đánh giá trên tập Test
```bash
python evaluate.py --decoder transformer
```

### Dự đoán nhanh (CLI)
```bash
python inference.py \
  --image data/sample_images/cam_thao.jpg \
  --question "Đây là cây gì?" \
  --decoder transformer
```

### Demo Web UI (Gradio)
```bash
python app.py --decoder transformer --port 7860
# Mở trình duyệt: http://localhost:7860
```

---

## 📁 Cấu trúc dự án

```
vqa-herb/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── train.py          # Entry point huấn luyện
├── evaluate.py       # Entry point đánh giá
├── inference.py      # Entry point dự đoán CLI
├── app.py            # Gradio Web UI Demo
│
├── configs/
│   └── config.py     # Hyperparameters & đường dẫn
│
├── data/
│   ├── vocab.py      # AnswerVocab
│   ├── dataset.py    # HerbVQADataset, DataLoader
│   └── sample_images/    # [TODO] Thêm 2-3 ảnh dược liệu mẫu
│
├── models/
│   ├── image_encoder.py      # ImageEncoder (DINOv2)
│   ├── text_encoder.py       # TextEncoder (PhoBERT)
│   ├── fusion.py             # CoAttentionFusion
│   ├── decoder_lstm.py       # LSTMDecoder (A1)
│   ├── decoder_transformer.py # TransformerDecoder (A2)
│   └── vqa_model.py          # VQAModel tổng thể
│
├── training/
│   ├── loss.py       # LabelSmoothedCrossEntropy
│   ├── optimizer.py  # build_optimizer_phase1/2
│   └── trainer.py    # train_one_epoch
│
├── evaluation/
│   └── metrics.py    # evaluate, compute_comprehensive_metrics
│
├── utils/
│   └── checkpoint.py # save/load checkpoint, plot_loss_curves
│
├── notebooks/
│   └── dl_final_vqa_vit_base_colab_DeepRL.ipynb  # Notebook gốc
│
└── docs/
    ├── images/           # [TODO] Sơ đồ kiến trúc, biểu đồ kết quả
    └── Bao_Cao_VQA.pdf   # [TODO] Báo cáo 15-20 trang
```

---

## 📊 Kết quả thực nghiệm

<!-- [TODO] Điền kết quả thực tế vào bảng dưới đây sau khi chạy evaluate.py -->

| Metric       | A1 – LSTM | A2 – Transformer |
|:-------------|:---------:|:----------------:|
| VQA Accuracy | ??.??%    | ??.??%           |
| BLEU-1       | 0.????    | 0.????           |
| BLEU-4       | 0.????    | 0.????           |
| ROUGE-L      | 0.????    | 0.????           |
| METEOR       | 0.????    | 0.????           |
| BERTScore F1 | 0.????    | 0.????           |

<!-- [TODO] Thêm biểu đồ so sánh
![Comparison Chart](docs/images/model_comparison.png)
-->

---

## 🖥 Demo Web UI

<!-- [TODO] Thêm ảnh chụp màn hình giao diện Gradio
![Gradio UI](docs/images/gradio_ui.png)
-->

<!-- [TODO] Thêm link video demo YouTube/Drive
📹 **Video demo (3-5 phút):** [CHÈN LINK YOUTUBE TẠI ĐÂY]
-->

---

## 📦 Dataset & Checkpoint

| Tài nguyên   | Link                                                                  |
|:-------------|:----------------------------------------------------------------------|
| Dataset      | [HuggingFace – azan100an/tdtu_vqa_dataset_herb](https://huggingface.co/datasets/azan100an/tdtu_vqa_dataset_herb) |
| Checkpoints  | [HuggingFace – azan100an/vqa-herb-checkpoints-Vit_base_colab](https://huggingface.co/azan100an/vqa-herb-checkpoints-Vit_base_colab) |

```bash
# Tải checkpoint tốt nhất về local
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='azan100an/vqa-herb-checkpoints-Vit_base_colab',
    filename='vqa_transformer_phase2_best.pt',
    local_dir='checkpoints/'
)
"
```

---

## 📝 Ghi chú cho người dùng

> **⚠️ TODO – Các việc cần bổ sung thủ công:**
> 1. Thêm 2-3 ảnh dược liệu mẫu vào `data/sample_images/` để demo.
> 2. Thêm ảnh sơ đồ kiến trúc vào `docs/images/architecture.png`.
> 3. Thêm ảnh chụp màn hình Gradio UI vào `docs/images/gradio_ui.png`.
> 4. Thêm ảnh biểu đồ kết quả vào `docs/images/model_comparison.png`.
> 5. Điền kết quả thực tế vào bảng **Kết quả thực nghiệm**.
> 6. Thêm link video demo YouTube/Google Drive.
> 7. Thêm file báo cáo PDF vào `docs/Bao_Cao_VQA.pdf`.
