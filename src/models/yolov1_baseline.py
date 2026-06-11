import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Class id -> human-readable name, used for per-class AP logging.
CLASS_NAMES = ["AirPlane", "Drone", "Helicopter"]


class YoloV1Baseline(L.LightningModule):
    """YOLOv1-style baseline detection model.
    
    Outputs a tensor of shape (B, 5 + num_classes, S, S).
    For each cell, it predicts:
    [x_offset, y_offset, w, h, objectness_logit, class_logits...]
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
        
        # 1x1 conv to map features to (5 + num_classes) per cell
        self.head = nn.Conv2d(backbone_out_channels, 5 + num_classes, kernel_size=1)
        self.lr = lr
        self.map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        features = self.backbone(x) # (B, C, S, S)
        out = self.head(features) # (B, 5 + num_classes, S, S)
        return out

    def _shared_loss(self, preds: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        B, _, S, _ = preds.shape
        
        # Split and apply sigmoids
        pred_xy = torch.sigmoid(preds[:, 0:2, :, :]) # (B, 2, S, S) - offsets [0, 1]
        pred_wh = torch.sigmoid(preds[:, 2:4, :, :]) # (B, 2, S, S) - normalized to image
        pred_obj = preds[:, 4:5, :, :]               # (B, 1, S, S) - logits
        pred_cls = preds[:, 5:, :, :]                # (B, C, S, S) - logits
        
        target_xy = torch.zeros_like(pred_xy)
        target_wh = torch.zeros_like(pred_wh)
        target_obj = torch.zeros_like(pred_obj)
        target_cls = torch.zeros_like(pred_cls)
        mask_obj = torch.zeros_like(pred_obj)
        
        # Assign targets
        for i in range(B):
            tgt = targets[i]
            if len(tgt) > 0:
                for obj in tgt:
                    gt_cls = int(obj[0].item())
                    cx, cy, w, h = obj[1:5]
                    
                    # Convert to grid coordinates
                    grid_x = int(cx.item() * S)
                    grid_y = int(cy.item() * S)
                    grid_x = max(0, min(grid_x, S - 1))
                    grid_y = max(0, min(grid_y, S - 1))
                    
                    # Only one object per cell
                    if mask_obj[i, 0, grid_y, grid_x] == 0:
                        mask_obj[i, 0, grid_y, grid_x] = 1.0
                        target_obj[i, 0, grid_y, grid_x] = 1.0
                        target_xy[i, 0, grid_y, grid_x] = cx * S - grid_x
                        target_xy[i, 1, grid_y, grid_x] = cy * S - grid_y
                        target_wh[i, 0, grid_y, grid_x] = w
                        target_wh[i, 1, grid_y, grid_x] = h
                        if 0 <= gt_cls < self.num_classes:
                            target_cls[i, gt_cls, grid_y, grid_x] = 1.0
                            
        mask_noobj = 1.0 - mask_obj
        
        # Objectness Loss with lambda_noobj
        loss_obj_pos = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='none') * mask_obj
        loss_obj_neg = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='none') * mask_noobj
        loss_obj = loss_obj_pos.sum() + 0.5 * loss_obj_neg.sum() # lambda_noobj = 0.5
        
        # Box Loss (Coord and Size) - only for responsible cells
        loss_xy = (F.mse_loss(pred_xy, target_xy, reduction='none') * mask_obj).sum()
        
        pred_wh_sqrt = torch.sqrt(pred_wh + 1e-6)
        target_wh_sqrt = torch.sqrt(target_wh + 1e-6)
        loss_wh = (F.mse_loss(pred_wh_sqrt, target_wh_sqrt, reduction='none') * mask_obj).sum()
        
        loss_box = 5.0 * (loss_xy + loss_wh) # lambda_coord = 5.0
        
        # Class Loss - only for responsible cells
        loss_cls = (F.binary_cross_entropy_with_logits(pred_cls, target_cls, reduction='none') * mask_obj).sum()
        
        num_objs = max(mask_obj.sum().item(), 1.0)
        
        return (loss_obj + loss_box + loss_cls) / num_objs

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        targets = batch["targets"]
        preds = self(images)
        
        loss = self._shared_loss(preds, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _decode_predictions(self, preds: torch.Tensor, img_shape: tuple[int, int],
                            objectness_threshold: float = 0.25) -> list[dict]:
        """Decode raw grid predictions into per-image detection dicts.

        Returns a list (length B) of {"boxes": (N,4) xyxy in pixels, "scores": (N,),
        "labels": (N,)} dicts — the format expected by torchmetrics and the
        visualization utilities. Shared by validation_step and the analysis notebook.
        """
        H_img, W_img = img_shape
        B, _, S, _ = preds.shape

        pred_xy = torch.sigmoid(preds[:, 0:2, :, :])
        pred_wh = torch.sigmoid(preds[:, 2:4, :, :])
        pred_obj = torch.sigmoid(preds[:, 4, :, :])   # (B, S, S)
        pred_cls = torch.sigmoid(preds[:, 5:, :, :])  # (B, C, S, S)

        preds_dict = []
        for i in range(B):
            mask = pred_obj[i] > objectness_threshold  # Threshold for objectness

            if mask.any():
                grid_y, grid_x = torch.where(mask)  # Cell coordinates

                valid_xy = pred_xy[i, :, grid_y, grid_x].T   # (N, 2) offsets
                valid_wh = pred_wh[i, :, grid_y, grid_x].T   # (N, 2) size
                valid_obj = pred_obj[i, grid_y, grid_x]      # (N,) objectness scores
                valid_cls = pred_cls[i, :, grid_y, grid_x].T  # (N, C) class probabilities

                cls_scores, cls_labels = torch.max(valid_cls, dim=1)  # (N,)
                final_scores = valid_obj * cls_scores  # Combined confidence

                # Decode relative (cell) coords back to global image coordinates
                cx = (grid_x.float() + valid_xy[:, 0]) / S
                cy = (grid_y.float() + valid_xy[:, 1]) / S
                w = valid_wh[:, 0]
                h = valid_wh[:, 1]

                xmin = (cx - w / 2) * W_img
                ymin = (cy - h / 2) * H_img
                xmax = (cx + w / 2) * W_img
                ymax = (cy + h / 2) * H_img

                valid_boxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

                preds_dict.append({
                    "boxes": valid_boxes_xyxy,
                    "scores": final_scores,
                    "labels": cls_labels,
                })
            else:
                preds_dict.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty((0,), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                })
        return preds_dict

    def _format_targets(self, targets: list[torch.Tensor], img_shape: tuple[int, int]) -> list[dict]:
        """Convert YOLO-normalized targets into xyxy pixel dicts for torchmetrics."""
        H_img, W_img = img_shape
        targets_dict = []
        for tgt in targets:
            if len(tgt) > 0:
                gt_boxes_xywh = tgt[:, 1:5]
                gt_boxes_xyxy = torch.zeros_like(gt_boxes_xywh)
                gt_boxes_xyxy[:, 0] = (gt_boxes_xywh[:, 0] - gt_boxes_xywh[:, 2] / 2) * W_img
                gt_boxes_xyxy[:, 1] = (gt_boxes_xywh[:, 1] - gt_boxes_xywh[:, 3] / 2) * H_img
                gt_boxes_xyxy[:, 2] = (gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2] / 2) * W_img
                gt_boxes_xyxy[:, 3] = (gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3] / 2) * H_img
                targets_dict.append({"boxes": gt_boxes_xyxy, "labels": tgt[:, 0].to(torch.int64)})
            else:
                targets_dict.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                })
        return targets_dict

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        targets = batch["targets"]
        preds = self(images)

        loss = self._shared_loss(preds, targets)
        self.log("val/loss", loss, prog_bar=True)

        img_shape = (images.shape[2], images.shape[3])
        preds_dict = self._decode_predictions(preds, img_shape)
        targets_dict = self._format_targets(targets, img_shape)
        self.map_metric.update(preds_dict, targets_dict)
        return loss

    def _log_map_metrics(self, prefix: str) -> None:
        """Compute, log (overall + per-class AP) and reset the mAP metric."""
        metrics = self.map_metric.compute()
        self.log(f"{prefix}/mAP50", metrics["map_50"], sync_dist=True)
        self.log(f"{prefix}/mAP50-95", metrics["map"], sync_dist=True)

        # class_metrics=True yields per-class AP, but only for classes present in the
        # split — metrics["classes"] gives the matching class ids (order is NOT a fixed
        # 0,1,2). Map each entry back to its name; clamp -1 (torchmetrics' "absent") to 0.
        per_class = metrics.get("map_per_class")
        class_ids = metrics.get("classes")
        if per_class is not None and per_class.ndim > 0 and class_ids is not None:
            for ap, cls_id in zip(per_class, class_ids.tolist()):
                if 0 <= cls_id < len(CLASS_NAMES):
                    self.log(f"{prefix}/AP_{CLASS_NAMES[cls_id]}", max(ap.item(), 0.0), sync_dist=True)

        self.map_metric.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_map_metrics("val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._log_map_metrics("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
