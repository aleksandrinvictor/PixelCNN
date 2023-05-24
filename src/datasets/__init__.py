from .mnist import MNIST
from .celeba import CelebA
from .cifar10 import CIFAR10

from typing import Tuple
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

__all__ = ["MNIST", "CelebA", "CIFAR10"]


def get_dataloaders_from_cfg(cfg: DictConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataloader = instantiate(cfg.dataset.train_dataloader)
    val_dataloader = instantiate(cfg.dataset.val_dataloader)

    return train_dataloader, val_dataloader
