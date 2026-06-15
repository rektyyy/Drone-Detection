import lightning as L
import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CLASS_NAMES = ["AirPlane", "Drone", "Helicopter"]


class TrivialBaseline(L.LightningModule):
    """Trivial detection baseline: always predict one fixed box per image.

    Ignores input completely. Uses dataset prior statistics:
    - prior_class: most frequent class in training set (Drone=1)
    - prior_w/h: mean bounding box dimensions from training labels

    Establishes a lower bound for mAP — any learned model should exceed this.
    """

    def __init__(
        self,
        prior_class: int = 1,
        prior_w: float = 0.089,
        prior_h: float = 0.088,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.prior_class = prior_class
        self.prior_w = prior_w
        self.prior_h = prior_h
        self.num_classes = num_classes
        # Dummy parameter so configure_optimizers has something to work with.
        self._dummy = nn.Parameter(torch.zeros(1))
        self.map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # No learning — return zero loss so the trainer completes without errors.
        return self._dummy.sum() * 0

    def _decode_predictions(self, images: torch.Tensor) -> list[dict]:
        """Return one fixed center box per image, ignoring pixel content."""
        B, _, H, W = images.shape
        cx, cy = W / 2.0, H / 2.0
        w, h = self.prior_w * W, self.prior_h * H
        box = torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]],
                           device=self.device)
        return [
            {
                "boxes": box.clone(),
                "scores": torch.ones(1, device=self.device),
                "labels": torch.tensor([self.prior_class], device=self.device),
            }
            for _ in range(B)
        ]

    def _format_targets(self, targets: list[torch.Tensor], img_shape: tuple[int, int]) -> list[dict]:
        H_img, W_img = img_shape
        targets_dict = []
        for tgt in targets:
            if len(tgt) > 0:
                gt = tgt[:, 1:5]
                xyxy = torch.stack([
                    (gt[:, 0] - gt[:, 2] / 2) * W_img,
                    (gt[:, 1] - gt[:, 3] / 2) * H_img,
                    (gt[:, 0] + gt[:, 2] / 2) * W_img,
                    (gt[:, 1] + gt[:, 3] / 2) * H_img,
                ], dim=1)
                targets_dict.append({"boxes": xyxy, "labels": tgt[:, 0].to(torch.int64)})
            else:
                targets_dict.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                })
        return targets_dict

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, targets = batch["images"], batch["targets"]
        img_shape = (images.shape[2], images.shape[3])
        preds = self._decode_predictions(images)
        tgts = self._format_targets(targets, img_shape)
        self.map_metric.update(preds, tgts)

    def _log_map_metrics(self, prefix: str) -> None:
        metrics = self.map_metric.compute()
        self.log(f"{prefix}/mAP50", metrics["map_50"], sync_dist=True)
        self.log(f"{prefix}/mAP50-95", metrics["map"], sync_dist=True)

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
        return torch.optim.Adam([self._dummy], lr=1e-3)
