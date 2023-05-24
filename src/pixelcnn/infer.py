from .model import PixelCNN
from typing import Optional, Dict
import torch
from tqdm import tqdm
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class InferConfig:
    n_channels: int
    out_channels: int
    n_filters: int
    n_layers: int
    image_height: int
    image_width: int


INFER_CONFIGS: Dict[str, InferConfig] = {
    "mnist": InferConfig(n_channels=1, out_channels=256, n_filters=128, n_layers=15, image_height=28, image_width=28),
    "cifar10": InferConfig(
        n_channels=3, out_channels=256 * 3, n_filters=128, n_layers=15, image_height=32, image_width=32
    ),
    "celeba": InferConfig(
        n_channels=3, out_channels=256 * 3, n_filters=128, n_layers=15, image_height=32, image_width=32
    ),
}


class InferModel:
    def __init__(
        self,
        infer_config: InferConfig,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
    ) -> None:
        self.infer_config = infer_config
        self.device = device

        self.model = PixelCNN(
            in_channels=infer_config.n_channels,
            out_channels=infer_config.out_channels,
            n_filters=infer_config.n_filters,
            n_layers=infer_config.n_layers,
        ).to(device)

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, num_images: int = 144) -> torch.Tensor:
        n_channels = self.infer_config.n_channels
        height = self.infer_config.image_height
        width = self.infer_config.image_width

        sample = torch.zeros((num_images, n_channels, height, width), device=self.device)
        sample.fill_(0)

        with tqdm(total=height * width * n_channels) as bar:
            for i in range(height):
                for j in range(width):
                    for k in range(n_channels):
                        logits = self.model.predict(sample)

                        logits = logits.view(num_images, 256, n_channels, height, width)
                        probs = F.softmax(logits[:, :, k, i, j], dim=1).data

                        labels = torch.multinomial(probs, 1).squeeze()

                        sample[:, k, i, j] = labels.float() / 255.0

                        bar.update(1)

        return sample

    @torch.no_grad()
    def autocomplete(self, image: torch.Tensor, num_images: int = 12) -> torch.Tensor:
        n_channels = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]

        sample = image.unsqueeze(0).repeat(num_images, 1, 1, 1)
        sample = sample.to(self.device)

        with tqdm(total=height * width * n_channels) as bar:
            for i in range(height):
                for j in range(width):
                    for k in range(n_channels):
                        if image[k, i, j] == -1:
                            logits = self.model.predict(sample)

                            logits = logits.view(num_images, 256, n_channels, height, width)
                            probs = F.softmax(logits[:, :, k, i, j], dim=1).data

                            labels = torch.multinomial(probs, 1).squeeze()

                            sample[:, k, i, j] = labels.float() / 255.0

                        bar.update(1)

        return sample
