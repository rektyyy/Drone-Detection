import torch
import torch.nn as nn

from src.models.modules.conv_block import ConvBlock


class ConvNetBackbone(nn.Module):
    """Small convolutional backbone for image inputs.

    Stacks a configurable number of Conv→BN→ReLU→MaxPool blocks, then
    flattens the spatial dimensions and projects to a dense embedding.

    Args:
        input_shape:   Shape of a single input sample (C, H, W).
        channel_dims:  Output channels for each conv block (e.g. [32, 64, 128]).
        output_dim:    Size of the backbone's output embedding.
        dropout:       Dropout probability before the final projection.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channel_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
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

        h, w = input_shape[1], input_shape[2]
        h_out = h // (2 ** len(channel_dims))
        w_out = w // (2 ** len(channel_dims))
        flat_dim = prev * h_out * w_out

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.projection = nn.Linear(flat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.projection(x)
