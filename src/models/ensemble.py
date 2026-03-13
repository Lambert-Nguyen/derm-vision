"""
ensemble.py
-----------
Weighted-averaging ensemble stub for the ISIC 2019 skin lesion classifier.

The ensemble combines the softmax outputs of multiple independently trained
models using a configurable per-model weight vector.

Usage
-----
    from models.ensemble import EnsembleModel
    from models.efficientnet import EfficientNetB3Classifier
    from models.custom_cnn import CustomCNN

    model_a = EfficientNetB3Classifier(num_classes=8)
    model_b = CustomCNN(num_classes=8)

    ensemble = EnsembleModel(
        models=[model_a, model_b],
        weights=[0.7, 0.3],
        num_classes=8,
    )
    logits = ensemble(images)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    """Weighted softmax-averaging ensemble over multiple classifiers.

    Parameters
    ----------
    models : list of nn.Module
        List of trained classification models.  Each model must accept an
        image tensor of shape ``(B, 3, H, W)`` and return logits of shape
        ``(B, num_classes)``.
    weights : list of float, optional
        Per-model mixing weights.  Will be L1-normalised to sum to 1.
        If ``None``, uniform weights are used.
    num_classes : int
        Number of output classes.  Default is 8.
    """

    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        weights: Optional[List[float]] = None,
        num_classes: int = 8,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.models = nn.ModuleList(models or [])

        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)
        else:
            n = len(self.models) if self.models else 1
            w = torch.ones(n, dtype=torch.float32)

        # Normalise so weights sum to 1
        w = w / w.sum()
        self.register_buffer("weights", w)

    def add_model(self, model: nn.Module, weight: float = 1.0) -> None:
        """Add a new model to the ensemble.

        Parameters
        ----------
        model : nn.Module
            Model to add.
        weight : float
            Mixing weight for the new model (will be re-normalised).
        """
        n_old = len(self.models)
        self.models.append(model)
        new_w = torch.cat(
            [self.weights * n_old, torch.tensor([weight])]
        )
        new_w = new_w / new_w.sum()
        self.register_buffer("weights", new_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted average of individual model probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``(B, num_classes)``.  Using
            log-probs ensures compatibility with ``nn.NLLLoss`` and
            ``torch.argmax``.

        Raises
        ------
        RuntimeError
            If the ensemble contains no models.
        """
        if len(self.models) == 0:
            raise RuntimeError(
                "The ensemble has no models.  Add at least one model before "
                "calling forward()."
            )

        weighted_probs: Optional[torch.Tensor] = None

        for model, w in zip(self.models, self.weights):
            probs = F.softmax(model(x), dim=1)
            if weighted_probs is None:
                weighted_probs = w * probs
            else:
                weighted_probs = weighted_probs + w * probs

        return torch.log(weighted_probs + 1e-8)  # log-probs
