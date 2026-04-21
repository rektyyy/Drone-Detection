import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationModel(L.LightningModule):
    """Lightning module for multi-class classification on a single attribute.

    Args:
        backbone:        Any nn.Module that maps (B, C, H, W) → (B, embed_dim).
        embed_dim:       Output size of the backbone.
        num_classes:     Number of target classes.
        attribute:       Key in the attributes dict to use as the label
                         (e.g. "gender", "race").
        lr:              Learning rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        attribute: str,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)
        self.attribute = attribute
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        images, attributes = batch
        labels = attributes[self.attribute]
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
