"""
efficientnet.py
---------------
EfficientNet-B3 transfer learning backbone for skin lesion classification.

Strategy
--------
1. Load ``efficientnet_b3`` pre-trained on ImageNet via ``timm``.
2. Replace the classification head with a new linear layer sized for
   ``num_classes`` output units.
3. Expose ``freeze_backbone()`` and ``unfreeze_backbone()`` helpers so that
   the caller can implement a two-stage fine-tuning strategy:

   * **Stage 1** – Freeze the convolutional backbone and train only the new
     head for a few epochs (fast convergence, avoids overwriting ImageNet
     features).
   * **Stage 2** – Unfreeze all (or selected) layers and fine-tune with a
     small learning rate.

Usage
-----
    model = EfficientNetB3Classifier(num_classes=8)
    model.freeze_backbone()   # Stage 1
    # ... train head only ...
    model.unfreeze_backbone() # Stage 2
    # ... fine-tune end-to-end ...
"""

import torch
import torch.nn as nn

try:
    import timm  # type: ignore
except ImportError as exc:
    raise ImportError(
        "The 'timm' package is required for EfficientNet.  "
        "Install it with:  pip install timm"
    ) from exc


class EfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 classifier with a custom head for *num_classes* outputs.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Default is 8 (ISIC 2019).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.  Default ``True``.
    dropout_rate : float
        Dropout probability applied inside the classification head.
        Default is 0.3.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Load backbone with timm (removes original classifier)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,  # remove original head
        )

        in_features = self.backbone.num_features  # 1536 for B3

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head remains trainable).

        Call this before Stage 1 training to speed up the initial head
        fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — only the classification head will be trained.")

    def unfreeze_backbone(self, layers_from_end: int = -1) -> None:
        """Unfreeze backbone parameters for end-to-end fine-tuning.

        Parameters
        ----------
        layers_from_end : int
            If >= 0, only unfreeze the last *layers_from_end* children of the
            backbone (e.g. ``4`` to unfreeze the last 4 blocks).
            If ``-1`` (default), all backbone parameters are unfrozen.
        """
        children = list(self.backbone.children())
        if layers_from_end == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Full backbone unfrozen for fine-tuning.")
        else:
            n = len(children)
            for i, child in enumerate(children):
                for param in child.parameters():
                    param.requires_grad = i >= n - layers_from_end
            print(f"Last {layers_from_end} backbone block(s) unfrozen for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.  EfficientNet-B3's native
            input resolution is 300×300.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        features = self.backbone(x)  # (B, 1536) after global pooling
        return self.head(features)
