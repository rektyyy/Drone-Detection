import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class SingleObjectYoloBaseline(L.LightningModule):
    """Baseline model simulating a YOLOv8-like output for a single object.
    
    Outputs a vector of length (4 + num_classes) per image:
    [cx, cy, w, h, logit_class_0, logit_class_1, ...]
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, 4 + num_classes)
        self.lr = lr
        self.map_metric = MeanAveragePrecision(box_format='xyxy')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features) # (B, 4 + num_classes)

    def _shared_loss(self, preds: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        B = preds.size(0)
        
        # Apply sigmoid only to bounding box coordinates to keep them in [0, 1]
        pred_boxes = torch.sigmoid(preds[:, :4]) 
        pred_logits = preds[:, 4:]
        
        loss_box = torch.tensor(0.0, device=self.device)
        loss_cls = torch.tensor(0.0, device=self.device)

        for i in range(B):
            tgt = targets[i]
            if len(tgt) > 0:
                first_gt = tgt[0]
                gt_cls = int(first_gt[0].item())
                gt_box = first_gt[1:5]
                
                # Box loss
                loss_box += F.mse_loss(pred_boxes[i], gt_box)
                
                # Class loss
                target_cls = torch.zeros(self.num_classes, device=self.device)
                if 0 <= gt_cls < self.num_classes:
                    target_cls[gt_cls] = 1.0
                loss_cls += F.binary_cross_entropy_with_logits(pred_logits[i], target_cls)
            else:
                # Image without objects - penalize class logits to predict no class
                target_cls = torch.zeros(self.num_classes, device=self.device)
                loss_cls += F.binary_cross_entropy_with_logits(pred_logits[i], target_cls)
                
        return (loss_box + loss_cls) / max(B, 1)

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
        
        # Update metrics
        B, _, H, W = images.shape
        preds_dict = []
        targets_dict = []
        
        for i in range(B):
            # Ground truth processing
            tgt = targets[i]
            if len(tgt) > 0:
                gt_boxes_xywh = tgt[:, 1:5]
                gt_boxes_xyxy = torch.zeros_like(gt_boxes_xywh)
                gt_boxes_xyxy[:, 0] = (gt_boxes_xywh[:, 0] - gt_boxes_xywh[:, 2] / 2) * W
                gt_boxes_xyxy[:, 1] = (gt_boxes_xywh[:, 1] - gt_boxes_xywh[:, 3] / 2) * H
                gt_boxes_xyxy[:, 2] = (gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2] / 2) * W
                gt_boxes_xyxy[:, 3] = (gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3] / 2) * H
                gt_labels = tgt[:, 0].to(torch.int64)
                targets_dict.append({"boxes": gt_boxes_xyxy, "labels": gt_labels})
            else:
                targets_dict.append({
                    "boxes": torch.empty((0, 4), device=self.device), 
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device)
                })
            
            # Prediction processing
            pred_box_xywh = torch.sigmoid(preds[i, :4])
            pred_logits = preds[i, 4:]
            pred_probs = torch.sigmoid(pred_logits)
            
            score, label = torch.max(pred_probs, dim=0)
            
            # Confidence threshold to simulate predicting "no object"
            if score > 0.01:
                cx, cy, w, h = pred_box_xywh
                xmin = (cx - w / 2) * W
                ymin = (cy - h / 2) * H
                xmax = (cx + w / 2) * W
                ymax = (cy + h / 2) * H
                pred_box_xyxy = torch.tensor([[xmin, ymin, xmax, ymax]], device=self.device)
                preds_dict.append({
                    "boxes": pred_box_xyxy,
                    "scores": torch.tensor([score], device=self.device),
                    "labels": torch.tensor([label], device=self.device)
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
        # Log explicitly map_50 and map. map is mAP@50:95
        self.log("val/mAP50", metrics["map_50"], sync_dist=True)
        self.log("val/mAP50-95", metrics["map"], sync_dist=True)
        self.map_metric.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
