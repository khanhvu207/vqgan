import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, **hparams):
        super().__init__()
        self.data_dir = hparams["data_dir"]
        self.batch_size = hparams["batch_size"]
        self.data_variance = 0.094930425
        self.name = "mnist"

    def prepare_data(self):
        # Lightning ensures the prepare_data() is called only within a single process on CPU
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        print("Data prepared!")

    def setup(self, stage=None):
        train_transforms = T.Compose(
            [
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        val_transforms = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        self.train_data = MNIST(
            self.data_dir, train=True, download=True, transform=train_transforms
        )
        self.val_data = MNIST(
            self.data_dir, train=False, download=True, transform=val_transforms
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return self.val_loader()
