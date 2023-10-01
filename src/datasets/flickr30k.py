import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

cv2.setNumThreads(
    0
)  # Somehow the training crashes with num_workers>0 without this line

import pytorch_lightning as pl

from functools import partial
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np

np.random.seed(42)

import os
import sys

sys.path.insert(0, "..")
from utils import *


class Flickr30k(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.image_dir = os.path.join(root, "flickr30k_images/flickr30k_images")
        self.caption_dir = os.path.join(root, "flickr30k_images/results.csv")
        self.num_images = len(os.listdir(self.image_dir))
        self.image_names = os.listdir(self.image_dir)
        self.transform = transform

        self.indices = np.arange(self.num_images)
        if train is True:
            self.indices = self.indices[: int(0.9 * self.num_images)]
        else:
            self.indices = self.indices[int(0.9 * self.num_images) :]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_name = self.image_names[self.indices[idx]]
        image = cv2.imread(os.path.join(self.image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
            image = inmap(image)

        return image


class Flickr30kDataModule(pl.LightningDataModule):
    def __init__(self, **conf):
        super().__init__()
        self.data_dir = "data"
        self.img_res = conf["img_res"]
        self.batch_size = conf["batch_size"]
        self.num_workers = conf["num_workers"]
        self.num_gpus = len(conf["gpus"])
        self.name = "flickr30k"

    def setup(self, stage=None):
        train_transforms = A.Compose(
            [
                A.SmallestMaxSize(self.img_res, interpolation=cv2.INTER_AREA),
                A.CenterCrop(self.img_res, self.img_res),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
        val_transforms = A.Compose(
            [
                A.SmallestMaxSize(self.img_res, interpolation=cv2.INTER_AREA),
                A.CenterCrop(self.img_res, self.img_res),
                ToTensorV2(),
            ]
        )
        self.train_data = Flickr30k(
            self.data_dir, train=True, transform=train_transforms
        )
        self.val_data = Flickr30k(self.data_dir, train=False, transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size // self.num_gpus,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_loader()
