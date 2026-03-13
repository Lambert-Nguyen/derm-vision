"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) visualisation stub
for the ISIC 2019 skin lesion classifier.

This module provides a thin wrapper around the ``grad-cam`` library
(``pip install grad-cam``) and a convenience function for overlaying the
heat-map on the original image.

References
----------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Usage
-----
    from gradcam import run_gradcam, save_gradcam_image
    heatmap, overlay = run_gradcam(model, image_tensor, target_layer)
    save_gradcam_image(overlay, "outputs/results/gradcam_sample.png")
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def run_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Grad-CAM heat-map for a single image.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.  Must have a convolutional backbone.
    input_tensor : torch.Tensor
        Pre-processed CHW float32 tensor (no batch dimension).
    target_layer : nn.Module
        The convolutional layer from which to extract activations (e.g.
        the last ``Conv2d`` before the global average pooling).
    target_class : int, optional
        Class index for which to compute the heat-map.  If ``None``, the
        predicted class (argmax) is used.
    device : torch.device, optional
        Compute device.  Inferred from *model* parameters if omitted.

    Returns
    -------
    heatmap : numpy.ndarray
        Normalised float32 heat-map of shape ``(H, W)`` in ``[0, 1]``.
    overlay : numpy.ndarray
        RGB uint8 overlay of the heat-map on the original image.

    Raises
    ------
    ImportError
        If the ``pytorch_grad_cam`` package is not installed.
    """
    try:
        from pytorch_grad_cam import GradCAM  # type: ignore
        from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'grad-cam' package is required for Grad-CAM visualisation. "
            "Install it with:  pip install grad-cam"
        ) from exc

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_batch = input_tensor.unsqueeze(0).to(device)

    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_batch, targets=targets)

    heatmap = grayscale_cam[0]  # shape: (H, W)

    # Build a [0,1] RGB version of the original image for overlay
    rgb_img = _tensor_to_rgb(input_tensor)
    overlay = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

    return heatmap, overlay


def save_gradcam_image(overlay: np.ndarray, save_path: str) -> None:
    """Save a Grad-CAM overlay image to disk.

    Parameters
    ----------
    overlay : numpy.ndarray
        RGB uint8 array of shape ``(H, W, 3)``.
    save_path : str
        Destination file path.  Parent directories are created if needed.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(save_path)
    print(f"Grad-CAM overlay saved to {save_path}")


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to a HWC float32 array in [0, 1].

    Uses approximate ImageNet denormalisation so that the overlay looks
    natural.

    Parameters
    ----------
    tensor : torch.Tensor
        CHW float32 tensor (normalised with ImageNet stats).

    Returns
    -------
    numpy.ndarray
        HWC float32 array with values in ``[0, 1]``.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = tensor.cpu().numpy().transpose(1, 2, 0)  # HWC
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick demo entry-point (stub)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml
    from transforms import get_val_test_transforms

    parser = argparse.ArgumentParser(description="Generate a Grad-CAM visualisation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--class_idx", type=int, default=None)
    parser.add_argument("--save", type=str, default="outputs/results/gradcam.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from train import build_model  # noqa: PLC0415

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load and preprocess the image
    transform = get_val_test_transforms(cfg.get("image_size", 300))
    raw = np.array(Image.open(args.image).convert("RGB"))
    result = transform(image=raw)
    img_tensor = result["image"]

    # Get the target conv layer (last block of EfficientNet-B3 backbone).
    # NOTE: This assumes the model is an EfficientNetB3Classifier instance.
    # For custom_cnn or ensemble models, pass the target layer explicitly.
    target_layer = model.backbone.features[-1]  # type: ignore[attr-defined]

    heatmap, overlay = run_gradcam(model, img_tensor, target_layer, args.class_idx, device)
    save_gradcam_image(overlay, args.save)
