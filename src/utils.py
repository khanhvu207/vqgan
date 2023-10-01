import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

inmap = lambda x: (x / 127.5 - 1.0).float()
outmap = lambda x: ((x + 1.0) * 127.5).clip(0, 255).long()


def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def cos_anneal(e0, e1, t0, t1, e):
    """ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1]"""
    alpha = max(
        0, min(1, (e - e0) / (e1 - e0))
    )  # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2)  # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0  # interpolate accordingly
    return t


def print_tensor_attributes(name, tensor):
    print(
        f"name: {name}:, "
        f"requires_grad: {tensor.requires_grad}, "
        f"grad_fn: {tensor.grad_fn is not None}, "
        f"is_leaf: {tensor.is_leaf}, "
        f"grad: {tensor.grad is not None}"
    )
