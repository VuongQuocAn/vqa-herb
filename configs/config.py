"""
configs/config.py — §2
Toàn bộ hyperparameters, đường dẫn, và cấu hình thiết bị cho dự án VQA Dược liệu.
"""
import os
import random
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace ──────────────────────────────────────────────────────────────
HF_DATASET_ID  = "azan100an/tdtu_vqa_dataset_herb"
HF_HUB_REPO_ID = "azan100an/vqa-herb-checkpoints-Vit_base_colab"
HF_HUB_TOKEN   = os.getenv("HF_TOKEN", "")

# ── Đường dẫn cục bộ ─────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
VOCAB_PATH     = os.path.join(BASE_DIR, "checkpoints", "vocab.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Tên trường dữ liệu ───────────────────────────────────────────────────────
FIELD_FILE_NAME = "file_name"
FIELD_QUESTION  = "question"
FIELD_ANSWER    = "answer"

# ── Token đặc biệt ───────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ── Tên model pretrained ──────────────────────────────────────────────────────
DINOV2_MODEL_NAME  = "facebook/dinov2-base"
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"

# ── Kiến trúc mô hình ────────────────────────────────────────────────────────
IMAGE_ENCODER_DIM      = 768
TEXT_ENCODER_DIM       = 768
PROJECTION_DIM         = 256
EMBEDDING_DIM          = 256

TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_NUM_HEADS  = 4
TRANSFORMER_FFN_DIM    = 512
TRANSFORMER_DROPOUT    = 0.3

LSTM_HIDDEN_DIM  = 512
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.3

MAX_QUESTION_LEN = 64
MAX_ANSWER_LEN   = 64
IMAGE_SIZE       = 224

# ── Tham số huấn luyện ───────────────────────────────────────────────────────
BATCH_SIZE            = 16
GRAD_ACCUMULATION     = 2
EPOCHS_PHASE1         = 10
EPOCHS_PHASE2         = 10
LR_PHASE1             = 1e-4
LR_PHASE2             = 5e-5
WARMUP_STEPS          = 150
LABEL_SMOOTHING       = 0.1
GRAD_CLIP_NORM        = 1.0
EARLY_STOPPING_PATIENCE = 4
NUM_WORKERS           = 2
SEED                  = 42

# ── Cấu hình thiết bị ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet normalization ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED) -> None:
    """Đặt seed toàn cục để đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
