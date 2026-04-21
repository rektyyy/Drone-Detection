import math

import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """Fully-connected backbone.

    Produces a fixed-size feature vector that a task head can consume.
    Accepts flat vectors or multi-dimensional inputs (e.g. images); the
    latter are flattened before the first linear layer.

    Args:
        input_shape: Shape of a single input sample, e.g. (784,) or (3, 32, 32).
        hidden_dims: Sizes of the hidden layers (e.g. [256, 128]).
        output_dim:  Size of the backbone's output embedding.
        dropout:     Dropout probability applied after every hidden activation.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        input_dim = math.prod(input_shape)
        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(inplace=True)]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))
