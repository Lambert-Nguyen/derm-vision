"""
app.py
------
Stub web application for the derm-vision skin lesion classifier.

This module is a placeholder for a future Gradio (or Flask) deployment.
It demonstrates how a trained EfficientNet-B3 model can be wrapped in an
interactive web UI that accepts a dermoscopic image and returns per-class
probabilities.

To run (once dependencies are installed and a checkpoint exists):

    python app/app.py --checkpoint outputs/checkpoints/best_model.pt

Dependencies
------------
    pip install gradio torch torchvision timm albumentations
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
CLASS_DESCRIPTIONS = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / Intraepithelial carcinoma",
    "BKL": "Benign keratosis-like lesion",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained EfficientNetB3Classifier from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint saved by ``train.py``.
    device : torch.device
        Compute device.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from models.efficientnet import EfficientNetB3Classifier  # noqa: PLC0415

    model = EfficientNetB3Classifier(num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def predict(
    model: torch.nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    image_size: int = 300,
) -> Dict[str, float]:
    """Run inference on a single PIL image and return a probability dict.

    Parameters
    ----------
    model : nn.Module
        Loaded classification model.
    pil_image : PIL.Image.Image
        Input dermoscopic image.
    device : torch.device
        Compute device.
    image_size : int
        Target resize dimension.  Should match the training config.

    Returns
    -------
    dict
        Mapping of ``class_name`` → ``probability`` (float in [0, 1]).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from transforms import get_val_test_transforms  # noqa: PLC0415

    transform = get_val_test_transforms(image_size)
    img_array = np.array(pil_image.convert("RGB"))
    tensor = transform(image=img_array)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


# ---------------------------------------------------------------------------
# Gradio interface (stub)
# ---------------------------------------------------------------------------


def build_gradio_app(model: torch.nn.Module, device: torch.device):
    """Build and return a Gradio interface for the classifier.

    Parameters
    ----------
    model : nn.Module
        Loaded classification model.
    device : torch.device
        Compute device.

    Returns
    -------
    gradio.Interface
        Gradio app (call ``.launch()`` to start the server).
    """
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Gradio is required for the web app.  Install with:  pip install gradio"
        ) from exc

    def inference_fn(pil_image: Image.Image) -> Dict[str, float]:
        """Gradio inference callback."""
        if pil_image is None:
            return {}
        return predict(model, pil_image, device)

    interface = gr.Interface(
        fn=inference_fn,
        inputs=gr.Image(type="pil", label="Upload dermoscopic image"),
        outputs=gr.Label(num_top_classes=len(CLASS_NAMES), label="Predicted probabilities"),
        title="DermVision — Skin Lesion Classifier",
        description=(
            "Upload a dermoscopic image to classify it into one of 8 ISIC 2019 "
            "skin lesion categories using an EfficientNet-B3 model.\n\n"
            + "\n".join(f"**{k}**: {v}" for k, v in CLASS_DESCRIPTIONS.items())
        ),
        examples=[],  # Add example image paths here once data is available
        allow_flagging="never",
    )
    return interface


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch the DermVision web app.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio server.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = load_model(args.checkpoint, _device)
    app = build_gradio_app(_model, _device)
    app.launch(server_port=args.port, share=args.share)
