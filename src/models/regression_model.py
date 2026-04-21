import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionModel(L.LightningModule):
    """Lightning module for scalar regression on a single attribute.

    Args:
        backbone:    Any nn.Module that maps (B, C, H, W) → (B, embed_dim).
        embed_dim:   Output size of the backbone.
        attribute:   Key in the attributes dict to use as the target
                     (e.g. "age").
        lr:          Learning rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        attribute: str,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, 1)
        self.attribute = attribute
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        images, attributes = batch
        targets = attributes[self.attribute].float()
        preds = self(images)
        loss = F.mse_loss(preds, targets)
        mae = (preds - targets).abs().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/mae", mae, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
