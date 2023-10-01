import gc
import os
from pprint import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from autoencoder import VQModel
from datasets import (CIFAR10DataModule, FFHQDataModule, Flickr30kDataModule,
                      MNISTDataModule)

device = "cuda:0"


def main(**args):
    assert args["run_id"] != "", "run_id is an empty string!"
    conf = OmegaConf.load(args["config"])
    pprint(conf)
    run_id = args["run_id"]
    debug = args["debug"]
    config_name = os.path.splitext(args["config"])[0].split("/")[1]
    log_dir = f"./logs/{config_name}/{run_id}"
    autoencoder = VQModel.load_from_checkpoint(os.path.join(log_dir, run_id + ".ckpt"))
    print(f"Stage 1 model loaded from {log_dir}")
    autoencoder.to(device)

    data = globals()[conf["dataset"]["target"]](**conf["train_1st_stage"])
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    train_quant_indices = []
    val_quant_indices = []
    with torch.inference_mode():
        for batch in tqdm(train_loader):
            quant, emb_loss, info = autoencoder.encode(batch.to(device))
            indices, _ = info
            indices = indices.cpu()
            train_quant_indices.append(indices)

        for batch in tqdm(val_loader):
            quant, emb_loss, info = autoencoder.encode(batch.to(device))
            indices, _ = info
            indices = indices.cpu()
            val_quant_indices.append(indices)

    train_tokens = torch.cat(train_quant_indices, dim=0)
    val_tokens = torch.cat(val_quant_indices, dim=0)
    print(train_tokens.shape, val_tokens.shape)
    torch.save((train_tokens, val_tokens), os.path.join(log_dir, "image_tokens.pt"))

    del autoencoder, data
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
