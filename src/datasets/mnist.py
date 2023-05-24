"""MNIST dataset."""

from typing import Any, Dict, Optional
from torch.utils.data.dataset import Dataset
import albumentations as A
import numpy as np
import torchvision


class MNIST(Dataset):
    def __init__(self, root: str, is_train: bool = True, transforms: Optional[A.Compose] = None) -> None:
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(root, train=is_train, download=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, label = self.dataset[index]

        image = np.array(image)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = image / 255.0

        return {"image": image.float(), "target": (image[0] * 255.0).long()}
