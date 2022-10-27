import json
from random import shuffle
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset


class TripletDataset(Dataset):
    def __init__(
        self,
        config,
    ):
        self.manifest_path = config.manifest_path
        self._load_meta()

    def _load_meta(self):
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        self.labels = set()
        self.index = dict()
        for i, d in enumerate(self.manifest):
            self.labels.add(d['class_name'])
            if d['class_name'] not in self.index:
                self.index[d['class_name']] = []
            self.index[d['class_name']].append(i)

    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        np.random.RandomState(1)

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.choice(self.index[self.manifest[idx]['class_name']])
        negative_label = np.random.choice(list(self.labels - {self.manifest[idx]['class_name']}))
        negative_idx = np.random.choice(self.index[negative_label])
        
        anchor_image = io.imread(self.manifest[idx]['image_path'])
        positive_image = io.imread(self.manifest[positive_idx]['image_path'])
        negative_image = io.imread(self.manifest[negative_idx]['image_path'])

        images = [anchor_image, positive_image, negative_image]
        for i in range(3):
            images[i] = torch.tensor(anchor_image)

        # for i, image in enumerate(images):
        #     if self.transform:
        #         images[i] = self.transform(image)

        return tuple(images), []


class TripletDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super(TripletDataModule, self).__init__()
        self.train_config = config.train
        self.val_config = config.val
        
    def setup(self, stage: Optional[str] = None):
        self.train_ds = TripletDataset(self.train_config)
        self.val_ds = TripletDataset(self.val_config)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=1,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        return [
            torch.stack(batch[:][i]) for i in range(3)
        ]
