"""CelebA dataset."""

from typing import Any, Dict, Optional
from torch.utils.data.dataset import Dataset
import albumentations as A
import torchvision
import numpy as np


class MyCelebA(torchvision.datasets.CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class CelebA(Dataset):
    def __init__(self, root: str, split: str = "train", transforms: Optional[A.Compose] = None) -> None:
        super().__init__()
        self.dataset = MyCelebA(root=root, split=split)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, label = self.dataset[index]

        image = np.array(image)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = image / 255.0

        return {"image": image, "target": (image * 255.0).long()}
