import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAgnosticDummyModel(L.LightningModule):
    """Class-agnostic dummy baseline model for YOLO format data.
    
    Uses a backbone to output a single embedding per image, and projects
    it to exactly 5 values: [confidence, x_center, y_center, width, height].
    
    The loss function only considers the FIRST ground truth box (if any)
    to calculate a very naive MSE loss for coordinates and BCE for confidence.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, 5)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        features = self.backbone(x)
        # output: (B, 5)
        return self.head(features)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        images = batch["images"]
        targets = batch["targets"]
        
        preds = self(images)
        loss = torch.tensor(0.0, device=self.device)
        
        for i in range(len(preds)):
            pred = preds[i] # (5,)
            tgt = targets[i] # (N, 5)
            
            # Use BCE for objectness/confidence logit
            if len(tgt) > 0:
                first_gt = tgt[0]
                gt_conf = torch.tensor(1.0, device=self.device)
                loss_conf = F.binary_cross_entropy_with_logits(pred[0], gt_conf)
                
                # Apply sigmoid so outputs are bounded in [0, 1] like the target coords
                pred_box = torch.sigmoid(pred[1:5])
                gt_box = first_gt[1:5]
                loss_box = F.mse_loss(pred_box, gt_box)
                
                loss += (loss_conf + loss_box)
            else:
                gt_conf = torch.tensor(0.0, device=self.device)
                loss_conf = F.binary_cross_entropy_with_logits(pred[0], gt_conf)
                loss += loss_conf
                
        loss = loss / max(len(preds), 1)
        
        self.log(f"{stage}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
