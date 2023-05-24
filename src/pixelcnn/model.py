from typing import Union, Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        mask_type: str = "A",
        stride: _size_2_t = 1,
        padding: Union[_size_2_t, str] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

        mask = torch.zeros_like(self.weight)

        # Spatial masking
        mask[:, :, : kernel_size // 2] = 1.0
        if mask_type == "A":
            mask[:, :, kernel_size // 2, : kernel_size // 2] = 1.0
        else:
            mask[:, :, kernel_size // 2, : kernel_size // 2 + 1] = 1.0

        # Channels masking [r, g, b, r, g, b, ...]
        if in_channels != 1:
            if mask_type == "A":
                mask_template = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
            else:
                mask_template = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

            n_out_tiles = out_channels // 3 + int(out_channels % 3 > 0)
            n_in_tiles = in_channels // 3 + int(in_channels % 3 > 0)

            channels_mask = torch.tile(mask_template, dims=(n_out_tiles, n_in_tiles))
            channels_mask = channels_mask[:out_channels, :in_channels]
            mask[:, :, kernel_size // 2, kernel_size // 2] = channels_mask

        self.register_buffer("mask", mask)

    def forward(self, x):
        with torch.no_grad():
            self.weight *= self.mask

        return super().forward(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.downsample = nn.Sequential(
            MaskedConv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                mask_type="B",
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.middle = nn.Sequential(
            MaskedConv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=3,
                mask_type="B",
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            MaskedConv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
                mask_type="B",
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.downsample(x)
        out = self.middle(out)
        out = self.upsample(out)

        return self.activation(out + x)


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        n_filters: int = 64,
        n_layers: int = 5,
        criterion: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(
            MaskedConv2d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=7,
                mask_type="A",
                padding="same",
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(n_filters))
        layers.append(nn.ReLU())

        for _ in range(n_layers):
            layers.append(ResBlock(in_channels=n_filters))

        layers.append(
            MaskedConv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1, mask_type="B", bias=False)
        )
        layers.append(nn.BatchNorm2d(n_filters))
        layers.append(nn.ReLU())
        layers.append(
            MaskedConv2d(
                in_channels=n_filters,
                out_channels=out_channels,
                kernel_size=1,
                mask_type="B",
            )
        )

        self.net = nn.Sequential(*layers)

        self.criterion = criterion

    def forward(self, image: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = self.net(image)

        output = {"logits": logits}

        if target is not None:
            batch_size, n_channels, height, width = image.shape
            loss = self.criterion(
                logits.view(batch_size, 256, n_channels, height, width),
                target.view(batch_size, n_channels, height, width),
            )
            output["loss"] = loss

        return output

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)
