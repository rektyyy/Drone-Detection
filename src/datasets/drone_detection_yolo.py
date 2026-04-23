from pathlib import Path

import lightning as L
import torch
import torchvision.transforms.functional as TF
import torchvision.tv_tensors as tv_tensors
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _parse_yolo_label_file(
    label_path: Path,
    orig_w: int,
    orig_h: int,
    out_size: int,
) -> torch.Tensor:
    """Parse YOLO txt lines into a (N, 5) tensor: ``cls, xc, yc, w, h`` normalized to ``out_size``.

    Each line: ``class_id x_center y_center width height`` with coordinates relative to the
    **original** image on disk. Boxes are scaled to match a stretched resize to
    ``(out_size, out_size)``.
    """
    if not label_path.is_file():
        return torch.zeros(0, 5, dtype=torch.float32)

    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return torch.zeros(0, 5, dtype=torch.float32)

    rows: list[list[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        c, xc, yc, w, h = map(float, parts[:5])
        x1 = (xc - w / 2) * orig_w
        x2 = (xc + w / 2) * orig_w
        y1 = (yc - h / 2) * orig_h
        y2 = (yc + h / 2) * orig_h
        x1s = x1 * out_size / orig_w
        x2s = x2 * out_size / orig_w
        y1s = y1 * out_size / orig_h
        y2s = y2 * out_size / orig_h
        wn = (x2s - x1s) / out_size
        hn = (y2s - y1s) / out_size
        xcn = (x1s + x2s) / 2 / out_size
        ycn = (y1s + y2s) / 2 / out_size
        rows.append([c, xcn, ycn, wn, hn])

    if not rows:
        return torch.zeros(0, 5, dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def collate_yolo_batch(batch: list[dict]) -> dict:
    """Stack images; keep targets as a list (variable object counts per image)."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["targets"] for b in batch]
    return {"images": images, "targets": targets}


class DroneDetectionYoloDataset(Dataset):
    """One YOLO split: ``split_dir/images`` + ``split_dir/labels`` (``*.txt`` per image)."""

    def __init__(
        self,
        split_dir: str | Path,
        img_size: int = 640,
    ) -> None:
        super().__init__()
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.img_size = img_size

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {self.labels_dir}")

        self._image_paths = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self._image_paths[idx]
        pil_image = Image.open(image_path).convert("RGB")
        img = tv_tensors.Image(TF.pil_to_tensor(pil_image))
        _, h0, w0 = img.shape
        img = TF.resize(img, [self.img_size, self.img_size])
        image = (img.float() / 255.0).clamp(0.0, 1.0)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        targets = _parse_yolo_label_file(label_path, w0, h0, self.img_size)
        return {"image": image, "targets": targets}


class DroneDetectionYoloDataModule(L.LightningDataModule):
    """YOLO-layout drone data: ``root/train``, ``root/valid``, ``root/test``.

    Each batch is ``{"images": (B, 3, H, W), "targets": List[Tensor]}`` where each list
    entry has shape ``(n_i, 5)`` with ``cls, xc, yc, w, h`` (YOLO-normalized to the
    resized square input).
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 8,
        img_size: int = 640,
        train_dir: str = "train",
        val_dir: str = "valid",
        test_dir: str = "test",
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        root = Path(root)
        self.train_dataset = DroneDetectionYoloDataset(root / train_dir, img_size=img_size)
        self.val_dataset = DroneDetectionYoloDataset(root / val_dir, img_size=img_size)
        self.test_dataset = DroneDetectionYoloDataset(root / test_dir, img_size=img_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_yolo_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_yolo_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_yolo_batch,
        )
