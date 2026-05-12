import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class GridYoloBaseline(L.LightningModule):
    """Baseline model for grid-based YOLO-like detection.
    
    Outputs a vector of shape (B, 4 + num_classes, num_predictions) where
    num_predictions = grid_size * grid_size.
    Each cell in the grid predicts [cx, cy, w, h, logit_cls0, logit_cls1, ...]
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_channels: int,
        grid_size: int = 8,
        num_classes: int = 3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        # 1x1 conv to map features to (4 box coords + num_classes) per cell
        self.head = nn.Conv2d(backbone_out_channels, 4 + num_classes, kernel_size=1)
        self.lr = lr
        self.map_metric = MeanAveragePrecision(box_format="xyxy")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        features = self.backbone(x) # (B, C, grid_size, grid_size)
        out = self.head(features) # (B, 4 + num_classes, grid_size, grid_size)
        
        B, C, H, W = out.shape
        # Flatten spatial dimensions -> (B, 4 + num_classes, Num_Predictions)
        return out.view(B, C, H * W)

    def _shared_loss(self, preds: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        B = preds.size(0)
        num_preds = self.grid_size * self.grid_size
        
        # pred_boxes: (B, 4, 64) - apply sigmoid to keep in [0, 1]
        pred_boxes = torch.sigmoid(preds[:, :4, :]) 
        # pred_logits: (B, 3, 64)
        pred_logits = preds[:, 4:, :] 
        
        target_boxes = torch.zeros_like(pred_boxes)
        target_cls = torch.zeros_like(pred_logits)
        mask = torch.zeros(B, num_preds, device=self.device)
        
        for i in range(B):
            tgt = targets[i]
            if len(tgt) > 0:
                for obj in tgt:
                    gt_cls = int(obj[0].item())
                    cx, cy, w, h = obj[1:5]
                    
                    # Find grid cell (0 to grid_size - 1)
                    grid_x = int(cx.item() * self.grid_size)
                    grid_y = int(cy.item() * self.grid_size)
                    
                    grid_x = max(0, min(grid_x, self.grid_size - 1))
                    grid_y = max(0, min(grid_y, self.grid_size - 1))
                    
                    cell_idx = grid_y * self.grid_size + grid_x
                    
                    mask[i, cell_idx] = 1.0
                    target_boxes[i, :, cell_idx] = obj[1:5]
                    if 0 <= gt_cls < self.num_classes:
                        target_cls[i, gt_cls, cell_idx] = 1.0
                        
        # Class loss (BCE over all cells, empty cells learn to predict 0 prob)
        loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_cls)
        
        # Box loss (MSE only for responsible cells)
        box_mse = F.mse_loss(pred_boxes, target_boxes, reduction="none") # (B, 4, 64)
        box_mse = box_mse * mask.unsqueeze(1) # Zero out non-responsible
        num_objs = mask.sum()
        
        loss_box = box_mse.sum() / max(num_objs, 1.0)
        
        # Sum with a weight multiplier for box to balance
        return loss_cls + 5.0 * loss_box

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        targets = batch["targets"]
        preds = self(images)
        
        loss = self._shared_loss(preds, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        targets = batch["targets"]
        preds = self(images)
        
        loss = self._shared_loss(preds, targets)
        self.log("val/loss", loss, prog_bar=True)
        
        # Metrics
        B = images.shape[0]
        img_h, img_w = images.shape[2], images.shape[3]
        preds_dict = []
        targets_dict = []
        
        pred_boxes = torch.sigmoid(preds[:, :4, :]) # (B, 4, 64)
        pred_probs = torch.sigmoid(preds[:, 4:, :]) # (B, 3, 64)
        
        for i in range(B):
            # Target processing
            tgt = targets[i]
            if len(tgt) > 0:
                gt_boxes_xywh = tgt[:, 1:5]
                gt_boxes_xyxy = torch.zeros_like(gt_boxes_xywh)
                gt_boxes_xyxy[:, 0] = (gt_boxes_xywh[:, 0] - gt_boxes_xywh[:, 2] / 2) * img_w
                gt_boxes_xyxy[:, 1] = (gt_boxes_xywh[:, 1] - gt_boxes_xywh[:, 3] / 2) * img_h
                gt_boxes_xyxy[:, 2] = (gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2] / 2) * img_w
                gt_boxes_xyxy[:, 3] = (gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3] / 2) * img_h
                gt_labels = tgt[:, 0].to(torch.int64)
                targets_dict.append({"boxes": gt_boxes_xyxy, "labels": gt_labels})
            else:
                targets_dict.append({
                    "boxes": torch.empty((0, 4), device=self.device), 
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device)
                })
                
            # Prediction processing
            scores, labels = torch.max(pred_probs[i], dim=0) # (64,)
            mask = scores > 0.25 # Only consider detections with >25% confidence
            
            if mask.any():
                valid_boxes_xywh = pred_boxes[i, :, mask].T # (N, 4)
                valid_scores = scores[mask]
                valid_labels = labels[mask]
                
                cx = valid_boxes_xywh[:, 0] * img_w
                cy = valid_boxes_xywh[:, 1] * img_h
                w = valid_boxes_xywh[:, 2] * img_w
                h = valid_boxes_xywh[:, 3] * img_h
                
                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2
                
                valid_boxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)
                preds_dict.append({
                    "boxes": valid_boxes_xyxy,
                    "scores": valid_scores,
                    "labels": valid_labels
                })
            else:
                preds_dict.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty((0,), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device)
                })
        
        self.map_metric.update(preds_dict, targets_dict)
        return loss
        
    def on_validation_epoch_end(self) -> None:
        metrics = self.map_metric.compute()
        self.log("val/mAP50", metrics["map_50"], sync_dist=True)
        self.log("val/mAP50-95", metrics["map"], sync_dist=True)
        self.map_metric.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
