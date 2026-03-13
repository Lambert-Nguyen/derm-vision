"""
evaluate.py
-----------
Evaluation utilities for the ISIC 2019 skin lesion classification task.

Provides
--------
* :func:`compute_metrics` — balanced accuracy, weighted F1, per-class
  precision/recall, and a confusion matrix.
* :func:`plot_confusion_matrix` — saves a labelled heatmap PNG.
* :func:`evaluate_model` — full evaluation loop over a DataLoader.

Usage
-----
    python evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import CLASS_NAMES


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : list of int
        Ground-truth class indices.
    y_pred : list of int
        Predicted class indices.
    class_names : list of str, optional
        Human-readable class names used in the per-class report.
        Defaults to ``CLASS_NAMES``.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"balanced_accuracy"`` (float)
        * ``"weighted_f1"`` (float)
        * ``"per_class_precision"`` (list of float, one per class)
        * ``"per_class_recall"`` (list of float, one per class)
        * ``"per_class_f1"`` (list of float, one per class)
        * ``"confusion_matrix"`` (2-D numpy array)
        * ``"classification_report"`` (str)
    """
    names = class_names or CLASS_NAMES
    labels = list(range(len(names)))

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    ).tolist()
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    ).tolist()
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    ).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, target_names=names, labels=labels, zero_division=0
    )

    return {
        "balanced_accuracy": bal_acc,
        "weighted_f1": weighted_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and optionally save a confusion-matrix heatmap.

    Parameters
    ----------
    cm : numpy.ndarray
        Square confusion matrix of shape ``(num_classes, num_classes)``.
    class_names : list of str, optional
        Tick labels.  Defaults to ``CLASS_NAMES``.
    save_path : str, optional
        If provided, the figure is saved to this path (PNG).
    title : str
        Figure title.
    normalize : bool
        Whether to display row-normalised (recall) values.  Default ``True``.
    """
    names = class_names or CLASS_NAMES

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_cm_path: Optional[str] = None,
) -> Dict:
    """Run full evaluation of *model* on *loader*.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate (in eval mode).
    loader : DataLoader
        Test / validation DataLoader.
    device : torch.device
        Compute device.
    class_names : list of str, optional
        Class names passed to :func:`compute_metrics`.
    save_cm_path : str, optional
        If provided, saves the confusion-matrix plot to this path.

    Returns
    -------
    dict
        Metrics dictionary from :func:`compute_metrics`.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)

    print("\n=== Evaluation Results ===")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        save_path=save_cm_path,
    )

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained skin lesion classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--save_cm",
        type=str,
        default="outputs/results/confusion_matrix.png",
        help="Path to save the confusion matrix image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import yaml
    from dataset import SkinLesionDataset
    from transforms import get_val_test_transforms

    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg.get("image_size", 300)

    dataset = SkinLesionDataset(
        img_dir=cfg["data"]["img_dir"],
        manifest_csv=cfg["data"].get(f"{args.split}_csv"),
        transform=get_val_test_transforms(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )

    # Lazy model import to avoid circular deps
    from train import build_model

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    evaluate_model(
        model,
        loader,
        device,
        class_names=cfg.get("class_names", CLASS_NAMES),
        save_cm_path=args.save_cm,
    )
