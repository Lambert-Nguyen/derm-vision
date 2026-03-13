"""
train.py
--------
Training loop for the ISIC 2019 skin lesion classification task.

Features
--------
* Weighted cross-entropy loss to handle class imbalance.
* Cosine-annealing learning-rate schedule with warm restarts.
* Weights & Biases (W&B) experiment tracking — pass ``--no_wandb`` to
  disable.
* Automatic best-checkpoint saving keyed on validation balanced accuracy.
* Configurable via a YAML file (see ``configs/config.yaml``) or CLI flags.

Usage
-----
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --no_wandb
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from dataset import SkinLesionDataset
from transforms import get_train_transforms, get_val_test_transforms
from evaluate import compute_metrics

# ---------------------------------------------------------------------------
# Optional W&B import — gracefully disabled if not installed / not desired
# ---------------------------------------------------------------------------
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model from the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.  The ``model`` key selects the
        architecture: ``"efficientnet"`` (default), ``"custom_cnn"``,
        or ``"ensemble"``.

    Returns
    -------
    nn.Module
        Instantiated model moved to the configured device.
    """
    model_name = cfg.get("model", "efficientnet")
    num_classes = cfg.get("num_classes", 8)

    if model_name == "efficientnet":
        from models.efficientnet import EfficientNetB3Classifier

        model = EfficientNetB3Classifier(num_classes=num_classes)
    elif model_name == "custom_cnn":
        from models.custom_cnn import CustomCNN

        model = CustomCNN(num_classes=num_classes)
    elif model_name == "ensemble":
        from models.ensemble import EnsembleModel

        model = EnsembleModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
) -> float:
    """Run a single training epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function (weighted cross-entropy).
    optimizer : optim.Optimizer
        Optimiser instance.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch index (0-based), used for logging.
    use_wandb : bool
        Whether to log per-step loss to W&B.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(loader):
        images, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if use_wandb and _WANDB_AVAILABLE:
            wandb.log({"train/step_loss": loss.item()})

    mean_loss = running_loss / len(loader)
    return mean_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate the model on a validation loader.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.

    Returns
    -------
    tuple
        ``(mean_loss, metrics_dict)`` where *metrics_dict* is produced by
        :func:`evaluate.compute_metrics`.
    """
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    mean_loss = running_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds)
    return mean_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
) -> None:
    """Save a training checkpoint to disk.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    optimizer : optim.Optimizer
        Optimiser (saved for resumable training).
    epoch : int
        Current epoch.
    metrics : dict
        Validation metrics at this epoch.
    save_path : str
        File path for the checkpoint (``*.pt``).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(cfg: dict, use_wandb: bool = True) -> None:
    """Run the full training pipeline.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (loaded from ``configs/config.yaml``).
    use_wandb : bool
        Whether to initialise a W&B run.  Falls back to console logging
        if W&B is unavailable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- W&B initialisation ----
    if use_wandb and _WANDB_AVAILABLE:
        wandb.init(
            project=cfg.get("wandb_project", "derm-vision"),
            config=cfg,
            name=cfg.get("run_name", None),
        )

    # ---- Datasets & loaders ----
    image_size = cfg.get("image_size", 300)
    train_dataset = SkinLesionDataset(
        img_dir=cfg["data"]["img_dir"],
        manifest_csv=cfg["data"].get("train_csv"),
        transform=get_train_transforms(image_size),
    )
    val_dataset = SkinLesionDataset(
        img_dir=cfg["data"]["img_dir"],
        manifest_csv=cfg["data"].get("val_csv"),
        transform=get_val_test_transforms(image_size),
    )

    class_weights = train_dataset.get_class_weights().to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # ---- Model, loss, optimiser, scheduler ----
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.get("scheduler_t0", 10),
        T_mult=cfg.get("scheduler_t_mult", 2),
    )

    best_bal_acc = 0.0
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "outputs/checkpoints"))

    # ---- Training loop ----
    for epoch in range(cfg.get("epochs", 30)):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_wandb
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        bal_acc = val_metrics["balanced_accuracy"]
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"bal_acc={bal_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if use_wandb and _WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        # Save best checkpoint
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            ckpt_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, str(ckpt_path))
            print(f"  → New best checkpoint saved (bal_acc={bal_acc:.4f})")

    # Save final checkpoint
    save_checkpoint(
        model,
        optimizer,
        cfg.get("epochs", 30) - 1,
        val_metrics,
        str(checkpoint_dir / "last_model.pt"),
    )

    if use_wandb and _WANDB_AVAILABLE:
        wandb.finish()

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a skin lesion classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, use_wandb=not args.no_wandb)
