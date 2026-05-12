import torch
import torch.nn as nn

from src.models.modules.conv_block import ConvBlock

class GridConvNetBackbone(nn.Module):
    """Small convolutional backbone for grid-based (YOLO-like) models.

    Stacks a configurable number of Conv→BN→ReLU→MaxPool blocks.
    Unlike ConvNetBackbone, it DOES NOT flatten the output, returning
    a 2D spatial feature map of shape (B, C, H_out, W_out).

    Args:
        input_shape:   Shape of a single input sample (C, H, W).
        channel_dims:  Output channels for each conv block (e.g. [16, 32, 64, 128]).
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channel_dims: list[int],
    ) -> None:
        super().__init__()

        conv_blocks: list[nn.Module] = []
        prev = input_shape[0]
        for out_ch in channel_dims:
            conv_blocks += [
                ConvBlock(prev, out_ch, kernel_size=3, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            prev = out_ch

        self.features = nn.Sequential(*conv_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
