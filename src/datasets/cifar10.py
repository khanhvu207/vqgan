import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import pytorch_lightning as pl

import sys

sys.path.insert(0, "..")
from utils import inmap, outmap


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, **conf):
        super().__init__()
        self.data_dir = "data"
        self.batch_size = conf["batch_size"]
        self.num_gpus = len(conf["gpus"])
        self.name = "cifar10"

    def prepare_data(self):
        # Lightning ensures the prepare_data() is called only within a single process on CPU
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
        print("Data prepared!")

    def setup(self, stage=None):
        train_transforms = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.PILToTensor(),
                inmap,
            ]
        )
        val_transforms = T.Compose(
            [
                T.PILToTensor(),
                inmap,
            ]
        )
        self.train_data = CIFAR10(
            self.data_dir, train=True, download=True, transform=train_transforms
        )
        self.val_data = CIFAR10(
            self.data_dir, train=False, download=True, transform=val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=32,
            num_workers=4,
            drop_last=True,
        )

    def test_dataloader(self):
        return self.val_loader()
