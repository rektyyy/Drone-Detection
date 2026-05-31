import lightning as L
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace


class UltralyticsDetectionModel(L.LightningModule):
    """Ultralytics YOLO model fine-tuned via a LightningModule.

    Expects batches from DroneDetectionYoloDataModule:
        {"images": (B, 3, H, W), "targets": List[Tensor(n_i, 5)]}
    where targets columns are: cls, cx, cy, w, h (YOLO-normalized).

    Args:
        model_name:  Full Ultralytics model name, e.g. "yolov8n", "yolo26n".
        num_classes: Number of detection classes.
        lr:          Learning rate.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        num_classes: int = 3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        yolo = YOLO(f"{model_name}.pt")
        self.model = yolo.model
        self.model.nc = num_classes

        # Ultralytics' loss criterion reads gain hyperparameters (box/cls/dfl) from
        # model.args. Loading just the bare model leaves these unset, so merge in defaults.
        args = dict(DEFAULT_CFG_DICT)
        if isinstance(self.model.args, dict):
            args.update(self.model.args)
        self.model.args = IterableSimpleNamespace(**args)

        # Pretrained weights ship frozen (requires_grad=False) — unfreeze for fine-tuning.
        for param in self.model.parameters():
            param.requires_grad = True

        self.map_metric = MeanAveragePrecision(box_format="xyxy")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _build_ultralytics_batch(
        self, images: torch.Tensor, targets: list[torch.Tensor]
    ) -> dict:
        device = images.device
        batch_idx_list, cls_list, bboxes_list = [], [], []
        for img_idx, tgt in enumerate(targets):
            if len(tgt) == 0:
                continue
            n = len(tgt)
            batch_idx_list.append(torch.full((n,), img_idx, dtype=torch.float32, device=device))
            cls_list.append(tgt[:, 0].to(device))
            bboxes_list.append(tgt[:, 1:5].to(device))

        if batch_idx_list:
            return {
                "batch_idx": torch.cat(batch_idx_list),
                "cls": torch.cat(cls_list).view(-1, 1),
                "bboxes": torch.cat(bboxes_list),
            }
        return {
            "batch_idx": torch.zeros(0, dtype=torch.float32, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        ut_batch = self._build_ultralytics_batch(images, batch["targets"])

        self.model.train()
        preds = self.model(images)
        loss, loss_items = self.model.loss(ut_batch, preds)

        self.log("train/loss", loss.sum(), prog_bar=True)
        self.log("train/loss_box", loss_items[0])
        self.log("train/loss_cls", loss_items[1])
        self.log("train/loss_dfl", loss_items[2])
        return loss.sum()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["images"]
        targets = batch["targets"]
        B, _, H, W = images.shape

        ut_batch = self._build_ultralytics_batch(images, targets)
        self.model.train()
        preds = self.model(images)
        loss, _ = self.model.loss(ut_batch, preds)
        self.log("val/loss", loss.sum(), prog_bar=True)

        self.model.eval()
        with torch.no_grad():
            raw = self.model(images)

        from ultralytics.utils.nms import non_max_suppression

        pred = raw[0] if isinstance(raw, (list, tuple)) else raw
        # End-to-end models (e.g. YOLO26) already return decoded detections as
        # (B, N, 6) = [x1, y1, x2, y2, conf, cls]. Classic models (e.g. YOLOv8)
        # return (B, 4+nc, anchors) which must be run through NMS.
        if pred.ndim == 3 and pred.shape[2] == 6:
            det_list = [d[d[:, 4] > 0.25] for d in pred]
        else:
            det_list = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        preds_list, targets_list = [], []
        for i in range(B):
            det = det_list[i]
            if det is not None and len(det):
                preds_list.append({
                    "boxes": det[:, :4],
                    "scores": det[:, 4],
                    "labels": det[:, 5].to(torch.int64),
                })
            else:
                preds_list.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty((0,), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                })

            tgt = targets[i]
            if len(tgt) > 0:
                cx, cy, w, h = tgt[:, 1], tgt[:, 2], tgt[:, 3], tgt[:, 4]
                gt_xyxy = torch.stack([
                    (cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H,
                ], dim=1)
                targets_list.append({"boxes": gt_xyxy, "labels": tgt[:, 0].to(torch.int64)})
            else:
                targets_list.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                })

        self.map_metric.update(preds_list, targets_list)

    def on_validation_epoch_end(self) -> None:
        metrics = self.map_metric.compute()
        self.log("val/mAP50", metrics["map_50"], sync_dist=True)
        self.log("val/mAP50-95", metrics["map"], sync_dist=True)
        self.map_metric.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
