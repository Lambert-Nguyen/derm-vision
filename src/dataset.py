"""
dataset.py
----------
PyTorch Dataset class for the ISIC 2019 skin lesion classification task.

Supports loading images from a directory (organised by class sub-folder or
from a flat CSV manifest) and optionally merging patient-level metadata.

Classes
-------
SkinLesionDataset
    Main dataset class used by all training and evaluation scripts.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_NAMES: List[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for ISIC 2019 skin lesion images.

    Expects either:
    * A CSV manifest with at minimum an ``image_id`` column and a label column
      (``dx`` or one of the 8 class-name columns with one-hot encoding), OR
    * A root directory whose sub-folders are the class names.

    Optionally merges a separate patient-metadata CSV keyed on ``image_id``.

    Parameters
    ----------
    img_dir : str or Path
        Root directory that contains the raw images (``<image_id>.jpg``).
    manifest_csv : str or Path, optional
        Path to a CSV file with image IDs and labels.  If ``None``, the
        dataset is built from sub-folder names inside *img_dir*.
    metadata_csv : str or Path, optional
        Path to a CSV file with additional per-image patient metadata
        (age, sex, localisation, etc.).  Merged on ``image_id``.
    transform : callable, optional
        Albumentations or torchvision transform applied to the image.
    class_names : list of str, optional
        Ordered list of class names.  Defaults to ``CLASS_NAMES``.
    use_metadata : bool
        Whether to return metadata features alongside the image tensor.
        Defaults to ``False``.
    img_ext : str
        Image file extension (default ``".jpg"``).
    """

    def __init__(
        self,
        img_dir: str,
        manifest_csv: Optional[str] = None,
        metadata_csv: Optional[str] = None,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        use_metadata: bool = False,
        img_ext: str = ".jpg",
    ) -> None:
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.use_metadata = use_metadata
        self.img_ext = img_ext

        self.samples: List[Tuple[Path, int]] = []
        self.metadata_df: Optional[pd.DataFrame] = None

        if manifest_csv is not None:
            self._load_from_csv(manifest_csv)
        else:
            self._load_from_folder()

        if metadata_csv is not None:
            self._load_metadata(metadata_csv)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_csv(self, manifest_csv: str) -> None:
        """Populate ``self.samples`` from a CSV manifest file.

        The CSV must contain an ``image_id`` column.  Labels are read from:
        * A ``dx`` column with string class names, OR
        * One-hot columns matching ``self.class_names`` (the argmax is used).
        """
        df = pd.read_csv(manifest_csv)
        if "dx" in df.columns:
            for _, row in df.iterrows():
                img_path = self.img_dir / (row["image_id"] + self.img_ext)
                label = self.class_to_idx[row["dx"]]
                self.samples.append((img_path, label))
        else:
            # Assume one-hot columns
            label_cols = [c for c in self.class_names if c in df.columns]
            for _, row in df.iterrows():
                img_path = self.img_dir / (row["image_id"] + self.img_ext)
                label = int(np.argmax([row[c] for c in label_cols]))
                self.samples.append((img_path, label))

    def _load_from_folder(self) -> None:
        """Populate ``self.samples`` by walking ``self.img_dir`` sub-folders."""
        for cls_name in self.class_names:
            cls_dir = self.img_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob(f"*{self.img_ext}")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

    def _load_metadata(self, metadata_csv: str) -> None:
        """Load and index patient metadata CSV keyed on ``image_id``."""
        df = pd.read_csv(metadata_csv)
        if "image_id" in df.columns:
            df = df.set_index("image_id")
        self.metadata_df = df

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int]:
        """Return a single (image_tensor, label) pair.

        If ``use_metadata`` is ``True`` and metadata was loaded, also returns
        a float tensor of metadata features as a third element.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        image : torch.Tensor
            CHW float32 tensor in [0, 1].
        label : int
            Integer class index.
        meta : torch.Tensor (only when ``use_metadata=True``)
            1-D float32 tensor of patient metadata features.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            # Support both albumentations dicts and torchvision callables
            if hasattr(self.transform, "transform"):
                augmented = self.transform(image=np.array(image))
                image = augmented["image"]
            else:
                image = self.transform(image)

        if self.use_metadata and self.metadata_df is not None:
            img_id = img_path.stem
            if img_id in self.metadata_df.index:
                meta_row = self.metadata_df.loc[img_id]
                meta_tensor = torch.tensor(
                    meta_row.values.astype(np.float32), dtype=torch.float32
                )
            else:
                meta_tensor = torch.zeros(
                    len(self.metadata_df.columns), dtype=torch.float32
                )
            return image, label, meta_tensor

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted cross-entropy.

        Returns
        -------
        torch.Tensor
            1-D float32 tensor of shape ``(num_classes,)``.
        """
        counts = torch.zeros(len(self.class_names))
        for _, label in self.samples:
            counts[label] += 1
        counts = counts.clamp(min=1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(self.class_names)
        return weights
