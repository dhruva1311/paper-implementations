"""Oxford-IIIT Pet foreground/background segmentation, resized for CPU training.

Trimaps (1 = pet, 2 = background, 3 = boundary) become binary masks with the
boundary counted as pet. Images are resized once and cached as uint8 tensors.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

BACKGROUND_TRIMAP_VALUE = 2


def _load_split(root: Path, split: str, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = OxfordIIITPet(str(root), split=split, target_types="segmentation", download=True)
    images = np.empty((len(raw), size, size, 3), dtype=np.uint8)
    masks = np.empty((len(raw), size, size), dtype=np.uint8)
    for i in tqdm(range(len(raw)), desc=f"Resizing {split} to {size}x{size}"):
        image, trimap = raw[i]
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
        trimap = trimap.resize((size, size), Image.NEAREST)
        images[i] = np.asarray(image)
        masks[i] = np.asarray(trimap) != BACKGROUND_TRIMAP_VALUE
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(masks)


def load_pets(root: str | Path = "data", size: int = 64) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return {"trainval": (images, masks), "test": (images, masks)} as uint8 tensors.

    Downloads the dataset into `root` on first use and caches the resized
    tensors at root/pets_{size}.pt.
    """
    root = Path(root)
    cache = root / f"pets_{size}.pt"
    if cache.exists():
        return torch.load(cache)
    root.mkdir(parents=True, exist_ok=True)
    data = {split: _load_split(root, split, size) for split in ("trainval", "test")}
    torch.save(data, cache)
    return data


class PetSegmentation(Dataset):
    """Wraps cached uint8 tensors; returns (float image in [0, 1], long mask).

    augment=True adds a random horizontal flip (the paper uses elastic
    deformations etc., Section 3.1, because it only has 30 images).
    """

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, augment: bool = False):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].float() / 255.0
        mask = self.masks[idx].long()
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            mask = mask.flip(-1)
        return image, mask
