import struct
from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torchvision.tv_tensors as tv_tensors
from torch.utils.data import DataLoader, Dataset, random_split


class MNISTDataset(Dataset):
    # label: digit class 0–9
    def __init__(
        self,
        images_path: str | Path,
        labels_path: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.transform = transform
        self._images = self._read_images(Path(images_path))
        self._labels = self._read_labels(Path(labels_path))
        assert len(self._images) == len(self._labels)

    @staticmethod
    def _read_images(path: Path) -> torch.Tensor:
        with open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">4I", f.read(16))
            assert magic == 2051, f"Unexpected magic number {magic} in {path}"
            data = torch.frombuffer(f.read(), dtype=torch.uint8)
        return data.reshape(n, 1, rows, cols)  # (N, 1, 28, 28)

    @staticmethod
    def _read_labels(path: Path) -> torch.Tensor:
        with open(path, "rb") as f:
            magic, n = struct.unpack(">2I", f.read(8))
            assert magic == 2049, f"Unexpected magic number {magic} in {path}"
            data = torch.frombuffer(f.read(), dtype=torch.uint8)
        return data.long()  # (N,)

    def __len__(self) -> int:
        return len(self._labels)

    @staticmethod
    def _normalize(img: torch.Tensor) -> torch.Tensor:
        return img.float() / 127.5 - 1.0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img = tv_tensors.Image(self._images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return self._normalize(img), {"label": self._labels[idx]}


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        data_dir = Path(data_dir)

        train_dataset = MNISTDataset(
            data_dir / "train-images.idx3-ubyte",
            data_dir / "train-labels.idx1-ubyte",
            transform=transform,
        )
        n_val = int(len(train_dataset) * val_split)
        n_train = len(train_dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(train_dataset, [n_train, n_val])

        self.test_dataset = MNISTDataset(
            data_dir / "t10k-images.idx3-ubyte",
            data_dir / "t10k-labels.idx1-ubyte",
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
