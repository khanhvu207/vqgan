import gc
import math
import os
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from autoencoder import VQModel
from generator import GenerativeModel
from utils import *

device = "cuda:0"


def main(**args):
    assert args["run_id"] != "", "run_id is an empty string!"
    conf = OmegaConf.load(args["config"])
    pprint(conf)
    run_id = args["run_id"]
    config_name = os.path.splitext(args["config"])[0].split("/")[1]
    log_dir = f"./logs/{config_name}/{run_id}"

    autoencoder = VQModel.load_from_checkpoint(os.path.join(log_dir, run_id + ".ckpt"))
    generator = GenerativeModel.load_from_checkpoint(
        os.path.join(log_dir, "generator_weights.ckpt"), stage_one_model=autoencoder
    ).to(device)
    generator.eval()

    print("Total number of parameters:", generator.count_parameters())

    img_res = conf["train_1st_stage"]["img_res"]
    f = len(conf["autoencoder"]["ch_mult"]) - 1
    token_length = int(img_res // (2**f)) ** 2

    c = torch.tensor([generator.sos_token], dtype=torch.long, device=device)
    num_img = 32
    print(f"Sampling {num_img} images...")
    start_time = time.time()
    with torch.inference_mode():
        generated_indices = generator.sample(
            batch_size=num_img,
            cond=c,
            length=token_length,
            top_k=100,
            top_p=None,
        )
        samples = generator.decode_to_img(generated_indices)
        samples = outmap(samples.cpu())
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Execution time: {exec_time:.4f} seconds")

    os.makedirs(log_dir + "/samples", exist_ok=True)
    for i in tqdm(range(num_img)):
        raw = samples[i].byte()
        image = Image.fromarray(raw.squeeze(0).permute(1, 2, 0).numpy(), "RGB")
        image = image.resize((512, 512), Image.BILINEAR)
        image.save(os.path.join(log_dir, f"samples/sample{i:02}.png"))

    fig, ax = plt.subplots(1, 1, dpi=300)
    ax.imshow(
        np.transpose(make_grid(samples, nrow=8).numpy(), (1, 2, 0)),
        interpolation="bilinear",
    )
    for axis in fig.axes:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"samples/sample_all.png"))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
