from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import torchvision.transforms as T


class DataModule(pl.LightningDataModule):
    _train_dataset: "_Dataset"
    _val_dataset: "_Dataset"

    def __init__(
            self,
            *,
            train_path: Path,
            val_path: Path,
            batch_size: int,
            num_workers: int,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def classes(self) -> List[int]:
        return self._train_dataset.classes

    @property
    def n_classes(self) -> int:
        return self._train_dataset.n_classes

    @property
    def dataset_size(self) -> int:
        return len(self._train_dataset)
        
    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = T.Compose([
            T.ToTensor(),
            #T.RandomRotation(180)
        ])
        self._train_dataset = _Dataset.from_folder(self.train_path, transform=train_transforms)
        self._val_dataset = _Dataset.from_folder(self.val_path, transform=T.ToTensor())
        

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._val_dataset, shuffle=False)


    def _create_dataloader(
            self,
            dataset: "_Dataset",
            *,
            shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class _Dataset(Dataset):
    def __init__(
            self,
            image_paths: List[Path],
            labels: List[int],
            transform,
    ):
        assert len(image_paths) == len(labels)
        assert len(image_paths) > 0, "data set is empty"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    @property
    def classes(self) -> List[int]:
        return sorted(list(np.unique(self.labels)))

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @staticmethod
    def from_folder(path: Path, transform) -> "_Dataset":
        image_paths = []
        labels = []

        for path in path.rglob("*.png"):
            image_paths.append(path)
            labels.append(int(path.parent.name))

        return _Dataset(image_paths=image_paths, labels=labels, transform=transform)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        label = self.labels[index]
        img = Image.open(self.image_paths[index])
        img = self.transform(img)

        return img, label

