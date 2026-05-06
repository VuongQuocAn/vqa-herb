"""utils/checkpoint.py — §10.5
save_checkpoint, load_checkpoint_if_exists, plot_loss_curves.
"""
import os
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, HfApi

from configs.config import CHECKPOINT_DIR, LOG_DIR, HF_HUB_REPO_ID, HF_HUB_TOKEN, DEVICE


def save_checkpoint(
    model, optimizer, scheduler,
    epoch, metrics,
    train_losses, val_losses,
    patience_count, path,
) -> None:
    """Lưu checkpoint xuống local và đẩy lên HuggingFace Hub."""
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = {
        "epoch":          epoch,
        "metrics":        metrics,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "decoder_type":   actual_model.decoder_type,
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "patience_count": patience_count,
    }
    torch.save(state, path)
    print(f"  Checkpoint lưu tại: {path}")

    if not HF_HUB_TOKEN or "your-username" in HF_HUB_REPO_ID:
        return
    try:
        api = HfApi()
        api.create_repo(repo_id=HF_HUB_REPO_ID, repo_type="model",
                        token=HF_HUB_TOKEN, exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=HF_HUB_REPO_ID,
            repo_type="model",
            token=HF_HUB_TOKEN,
        )
        print(f"  Đã đẩy lên HuggingFace Hub: {HF_HUB_REPO_ID}")
    except Exception as e:
        print(f"  Lỗi đẩy HF Hub: {e}")


def load_checkpoint_if_exists(
    filename, model, optimizer=None, scheduler=None,
    repo_id=HF_HUB_REPO_ID, device=DEVICE,
):
    """
    Kéo checkpoint từ HF (nếu có) và nạp trạng thái để resume.
    Trả về: (start_epoch, train_losses, val_losses, best_metric, patience_count)
    """
    local_path = os.path.join(CHECKPOINT_DIR, filename)

    if HF_HUB_TOKEN and "your-username" not in repo_id:
        try:
            print(f"Tìm '{filename}' trên HuggingFace Hub...")
            local_path = hf_hub_download(
                repo_id=repo_id, filename=filename,
                local_dir=CHECKPOINT_DIR, token=HF_HUB_TOKEN,
            )
            print("  -> Kéo thành công từ HF!")
        except Exception:
            print("  -> Không có trên HF hoặc lỗi mạng.")

    if not os.path.exists(local_path):
        print("  -> Chưa có checkpoint. Bắt đầu từ Epoch 1.")
        return 1, [], [], float("inf"), 0

    print(f"Nạp checkpoint từ: {local_path} ...")
    ckpt = torch.load(local_path, map_location=device)

    state_dict     = ckpt["model"]
    new_state_dict = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )
    model.load_state_dict(new_state_dict)

    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("  -> Bỏ qua nạp Optimizer do cấu trúc thay đổi.")

    if scheduler and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass

    start_epoch    = ckpt.get("epoch", 0) + 1
    train_losses   = ckpt.get("train_losses", [])
    val_losses     = ckpt.get("val_losses",   [])
    best_metric    = ckpt.get("metrics", {}).get("val_loss", float("inf"))
    patience_count = ckpt.get("patience_count", 0)

    print(f"  -> Thành công! Tiếp tục từ Epoch {start_epoch}.")
    return start_epoch, train_losses, val_losses, best_metric, patience_count


def plot_loss_curves(train_losses: list, val_losses: list, phase: int) -> None:
    """Vẽ biểu đồ Loss train và Loss val theo epoch."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, marker="o", color="steelblue", label="Train Loss")
    ax1.set_title(f"Train Loss — Giai đoạn {phase}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, val_losses, marker="s", color="darkorange", label="Val Loss")
    ax2.set_title(f"Validation Loss — Giai đoạn {phase}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, f"phase{phase}_loss_curves.png")
    plt.savefig(out_path, dpi=100)
    plt.show()
    print(f"  Biểu đồ lưu tại: {out_path}")
