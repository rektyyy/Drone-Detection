import torch.nn as nn


class ConvBlock(nn.Sequential):
    """Conv → BN → ReLU helper."""

    def __init__(self, in_channels: int, out_channels: int, **conv_kwargs) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
