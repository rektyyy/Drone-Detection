from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torchvision.transforms.functional as TF
import torchvision.tv_tensors as tv_tensors
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class UTKFaceDataset(Dataset):
    # gender: 0 = Male, 1 = Female
    # race: 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Other
    def __init__(
        self,
        data_dir: str | Path,
        size: int = 32,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.size = size
        self.transform = transform

        self._image_list = [
            path for path in self.data_dir.glob("*.jpg")
            if self._is_valid_filename(path)
        ]

    def __len__(self) -> int:
        return len(self._image_list)

    def _load_pil(self, idx: int) -> Image.Image:
        image_path = self._image_list[idx]
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _is_valid_filename(path: Path) -> bool:
        parts = path.stem.split("_")
        if len(parts) < 4:
            return False
        try:
            int(parts[0])
            int(parts[1])
            int(parts[2])
        except ValueError:
            return False
        return True

    @staticmethod
    def _normalize(img: torch.Tensor) -> torch.Tensor:
        return img.float() / 127.5 - 1.0

    def _parse_labels(self, idx: int) -> dict:
        stem = self._image_list[idx].stem  # e.g. "25_0_2_20170116174525125"
        parts = stem.split("_")
        return {
            "age": int(parts[0]),
            "gender": int(parts[1]),
            "race": int(parts[2]),
        }

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        pil_image = self._load_pil(idx)
        img = tv_tensors.Image(TF.pil_to_tensor(pil_image))
        img = TF.resize(img, [self.size, self.size])
        if self.transform is not None:
            img = self.transform(img)
        return self._normalize(img), self._parse_labels(idx)


class UTKFaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        size: int = 32,
        transform: Callable[[Image.Image], Image.Image] | None = None,
        val_split: float = 0.1,
        test_split: float = 0.2,
    ):
        super().__init__()

        self.batch_size = batch_size

        full_dataset = UTKFaceDataset(data_dir, size=size, transform=transform)
        n = len(full_dataset)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
