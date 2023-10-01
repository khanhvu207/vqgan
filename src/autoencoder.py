import datetime
import math
import os
from functools import partial
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import wandb
from einops import rearrange, reduce
from matplotlib.ticker import MultipleLocator, PercentFormatter
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchview import draw_graph
from torchvision.utils import make_grid

from datasets import (CIFAR10DataModule, FFHQDataModule, Flickr30kDataModule,
                      MNISTDataModule)
from modules.blocks import Conv2d as PatchedConv2d
from modules.blocks import Decoder, Encoder, OpenAIDecoder, OpenAIEncoder
from modules.loss import VQLPIPS, VQLPIPSWithDiscriminator
from modules.quantization import EMAQuantize, STQuantize
from utils import *


class VQModel(pl.LightningModule):
    def __init__(self, **conf):
        super().__init__()
        self.save_hyperparameters()
        self.log_dir = conf["log_dir"]

        self.ae_conf = conf["autoencoder"]
        self.encoder = Encoder(**self.ae_conf)

        self.vq_conf = conf["vector_quantization"]
        self.quant_conv = nn.Conv2d(
            self.ae_conf["z_channels"], self.vq_conf["params"]["code_dim"], 1
        )
        self.post_quant_conv = nn.Conv2d(
            self.vq_conf["params"]["code_dim"], self.ae_conf["z_channels"], 1
        )

        make_quantizer = partial(globals()[self.vq_conf["quantizer"]])
        self.quantize = make_quantizer(**self.vq_conf["params"])

        self.decoder = Decoder(**self.ae_conf)
        self.train_conf = conf["train_1st_stage"]
        self.data_conf = conf["dataset"]

        self.loss_fn = globals()[self.train_conf["loss_fn"]["target"]](
            **self.train_conf["loss_fn"]["params"]
        )

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.gradient_accumulation_steps = self.train_conf["grad_accum"]
        self.log_every = self.train_conf["log_every"]
        self.vq_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.quantize.parameters())
            + list(self.post_quant_conv.parameters())
        )
        self.disc_params = self.loss_fn.discriminator.parameters()
        self.code_norms = []
        self.code_norm_min = 100
        self.code_norm_max = 0
        self.train_steps = []
        self.current_step = 0

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, emb_loss, info = self.encode(input)
        dec = self.decode(quant)
        return dec, emb_loss, info

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def training_step(self, batch, batch_idx):
        x = batch
        vq_optimizer, disc_optimizer = self.optimizers()
        x_recon, emb_loss, info = self(x)
        _, perplexity = info
        self.log("train/perplexity", perplexity.item())

        self.toggle_optimizer(vq_optimizer)
        g_loss, g_loss_items = self.loss_fn(
            codebook_loss=emb_loss,
            inputs=x,
            reconstructions=x_recon,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            cond=None,
            split="train",
        )
        self.manual_backward(g_loss)
        if self.current_step % self.log_every == 0:
            self.log_grad()
        vq_optimizer.step()
        vq_optimizer.zero_grad()
        self.untoggle_optimizer(vq_optimizer)

        self.toggle_optimizer(disc_optimizer)
        d_loss, d_loss_items = self.loss_fn(
            codebook_loss=emb_loss,
            inputs=x,
            reconstructions=x_recon,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            cond=None,
            split="train",
        )
        self.manual_backward(d_loss)
        disc_optimizer.step()
        disc_optimizer.zero_grad()
        self.untoggle_optimizer(disc_optimizer)

        self.current_step += 1

        loss_items = merge_dicts(g_loss_items, d_loss_items)
        for loss_name, loss_value in loss_items.items():
            self.log(f"{loss_name}", loss_value)

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, emb_loss, info = self(x)
        _, perplexity = info
        self.log(
            "val/perplexity",
            perplexity.item(),
            sync_dist=True,
        )

        g_loss, g_loss_items = self.loss_fn(
            codebook_loss=emb_loss,
            inputs=x,
            reconstructions=x_recon,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            cond=None,
            split="val",
        )

        d_loss, d_loss_items = self.loss_fn(
            codebook_loss=emb_loss,
            inputs=x,
            reconstructions=x_recon,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            cond=None,
            split="val",
        )

        loss_items = merge_dicts(g_loss_items, d_loss_items)
        for loss_name, loss_value in loss_items.items():
            self.log(f"{loss_name}", loss_value, sync_dist=True)

    def log_grad(self):
        """
        Visualize gradients of the following modules:
            - self.quantize.embedding
        """
        assert (
            self.quantize.embedding.weight.grad is not None
        ), "Codes do not receive any gradients!"
        code_jacobi = self.quantize.embedding.weight.grad.clone()
        accum_jacobi = self.all_gather(code_jacobi)  # Gather across GPUs

        if self.trainer.is_global_zero is False:
            return

        accum_jacobi = accum_jacobi.mean(dim=0)
        grad_norm = torch.linalg.vector_norm(accum_jacobi, dim=1)
        grad_norm = grad_norm.cpu().numpy()
        code_norm = torch.linalg.vector_norm(
            self.quantize.embedding.weight.data.clone(), dim=1
        )
        code_norm = code_norm.cpu().numpy()
        self.code_norms.append(code_norm)
        self.code_norm_min = min(self.code_norm_min, code_norm.min())
        self.code_norm_max = max(self.code_norm_max, code_norm.max())
        self.train_steps.append(self.global_step // 2)

        # Gradient norm histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
        ax.hist(grad_norm, bins=100)
        ax.set_xlabel("Gradient norm", fontsize=12)
        ax.set_ylabel("Number of codes", fontsize=12)
        ax.grid(visible=True)
        plt.tight_layout()
        hist_file_name = f"Step{self.global_step:06}-code_grad_norm_hist.png"
        os.makedirs(self.log_dir + "/gradients", exist_ok=True)
        hist_img_save_path = os.path.join(self.log_dir + "/gradients", hist_file_name)
        plt.savefig(hist_img_save_path)
        plt.clf()

        # Code norm histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
        ax.hist(code_norm, bins=100)
        ax.set_xlabel("Code norm", fontsize=12)
        ax.set_ylabel("Number of codes", fontsize=12)
        ax.grid(visible=True)
        plt.tight_layout()
        code_norm_file_name = f"Step{self.global_step:06}-code_norm_hist.png"
        os.makedirs(self.log_dir + "/gradients", exist_ok=True)
        code_norm_img_save_path = os.path.join(
            self.log_dir + "/gradients", code_norm_file_name
        )
        plt.savefig(code_norm_img_save_path)
        plt.clf()

        # Code norm time-series histogram
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
        histograms = []

        for t, data in enumerate(self.code_norms):
            hist, bins = np.histogram(
                data, bins=100, range=(self.code_norm_min, self.code_norm_max)
            )
            histograms.append(hist)

        histograms = np.array(histograms)
        t = histograms.shape[0]
        cax = ax.matshow(
            np.flipud(histograms.T),
            cmap="plasma",
            aspect="auto",
            extent=[0, t - 1, self.code_norm_min, self.code_norm_max],
        )
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("Code norm", fontsize=12)
        ax.set_xticks(np.arange(t - 1) + 0.5)
        ax.set_xticklabels(self.train_steps[:-1])
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        cbar = fig.colorbar(
            cax,
            format=PercentFormatter(
                xmax=self.vq_conf["params"]["num_codes"], decimals=1
            ),
            pad=0,
        )
        cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom", fontsize=12)
        plt.tight_layout()
        ts_code_norm_file_name = f"Step{self.global_step:06}-ts_code_norm_hist.png"
        os.makedirs(self.log_dir + "/gradients", exist_ok=True)
        ts_code_norm_img_save_path = os.path.join(
            self.log_dir + "/gradients", ts_code_norm_file_name
        )
        plt.savefig(ts_code_norm_img_save_path)
        plt.clf()

        # Gradient norm heatmap
        fix, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        sz = int(np.sqrt(grad_norm.shape[0]))
        grad_map = np.reshape(grad_norm, (sz, sz))
        im = ax.imshow(grad_map, cmap="plasma")

        # Adjust the colorbar position and label
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Gradient norm", rotation=-90, va="bottom", fontsize=12)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Add a visible white grid
        ax.set_xticks(np.arange(-0.5, sz, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, sz, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.75)

        # Annotate each cell with indices
        threshold = grad_map.max() / 2.0
        for i in range(sz):
            for j in range(sz):
                textcolor = "k" if grad_map[i, j] > threshold else "w"
                text = ax.text(
                    j,
                    i,
                    str(i * sz + j),
                    ha="center",
                    va="center",
                    color=textcolor,
                    fontsize=5,
                )

        # Hide major ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.tick_params(which="minor", bottom=False, left=False)

        heatmap_file_name = f"Step{self.global_step:06}-code_grad_norm_map.png"
        os.makedirs(self.log_dir + "/gradients", exist_ok=True)
        heatmap_img_save_path = os.path.join(
            self.log_dir + "/gradients", heatmap_file_name
        )
        plt.savefig(heatmap_img_save_path)
        plt.clf()

        if self.logger is not None:
            self.logger.experiment.log(
                {
                    "Train statistics": [
                        wandb.Image(
                            hist_img_save_path,
                            caption=f"Codebook gradient histogram, Step {self.global_step:06}",
                        ),
                        wandb.Image(
                            heatmap_img_save_path,
                            caption=f"Codebook gradient heatmap, Step {self.global_step:06}",
                        ),
                        wandb.Image(
                            code_norm_img_save_path,
                            caption=f"Codebook vector norm, Step {self.global_step:06}",
                        ),
                        wandb.Image(
                            ts_code_norm_img_save_path,
                            caption=f"Codebook vector norm (Time series), Step {self.global_step:06}",
                        ),
                    ]
                },
                step=self.global_step,
            )

        plt.close()

    @torch.inference_mode()
    def on_validation_end(self):
        if self.trainer.is_global_zero is False:
            return

        val_ds = self.trainer.datamodule.val_data
        x = torch.cat([val_ds[idx].unsqueeze(0) for idx in range(4)], dim=0)
        dec, _, _ = self(x.to(self.device))
        x_recon = make_grid(outmap(dec.cpu()))
        x = make_grid(outmap(x))

        # Visualize reconstruction
        fig, ax = plt.subplots(2, 1, dpi=150)
        ax[0].imshow(np.transpose(x.numpy(), (1, 2, 0)), interpolation="nearest")
        ax[1].imshow(np.transpose(x_recon.numpy(), (1, 2, 0)), interpolation="nearest")
        for axis in fig.axes:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        plt.tight_layout()

        file_name = f"{self.current_epoch:04}-recon.png"
        os.makedirs(self.log_dir + "/reconstruction", exist_ok=True)
        img_save_path = os.path.join(self.log_dir + "/reconstruction", file_name)
        plt.savefig(img_save_path)
        plt.clf()

        if self.logger is not None:
            self.logger.experiment.log(
                {
                    "Reconstruction": [
                        wandb.Image(
                            img_save_path,
                            caption=f"Reconstruction, Epoch {self.current_epoch}",
                        ),
                    ]
                },
                step=self.global_step,
            )
        plt.close()

    def configure_optimizers(self):
        vq_optimizer = torch.optim.Adam(
            self.vq_params,
            lr=self.train_conf["lr"],
            betas=(0.5, 0.9),
        )
        disc_optimizer = torch.optim.Adam(
            self.disc_params,
            lr=self.train_conf["lr"],
            betas=(0.5, 0.9),
        )

        return vq_optimizer, disc_optimizer

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def on_train_batch_start(self, batch, batch_idx):
        warmup_step_nr = int(self.train_conf["lr_warmup"] * self.trainer.max_steps)
        current_step = self.global_step
        lr = 0
        for optimizer in self.optimizers():
            for g in optimizer.param_groups:
                if current_step < warmup_step_nr:
                    g["lr"] = current_step / warmup_step_nr * self.train_conf["lr"]
                else:
                    g["lr"] = cos_anneal(
                        warmup_step_nr,
                        self.trainer.max_steps,
                        self.train_conf["lr"],
                        self.train_conf["min_lr"],
                        self.global_step,
                    )
                lr = max(lr, g["lr"])

        self.log("train/lr", lr)


def main(**args):
    assert args["run_id"] != "", "run_id is an empty string!"
    conf = OmegaConf.load(args["config"])
    pprint(conf)

    run_id = args["run_id"]
    debug = args["debug"]
    data = globals()[conf["dataset"]["target"]](**conf["train_1st_stage"])

    config_name = os.path.splitext(args["config"])[0].split("/")[1]
    log_dir = f"./logs/{config_name}/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    logger = WandbLogger(
        project="generative",
        entity="kvu207",
        name=f"[1]{run_id}",
        save_dir=log_dir,
        offline=debug,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=log_dir,
        filename=f"{run_id}",
        save_top_k=1,
        mode="min",
    )

    model = VQModel(log_dir=log_dir, **conf)
    print(model)

    callback_list = [checkpoint_callback] if not debug else []

    # Track gradients and weights
    # wandb.watch(model, log='all')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=conf["train_1st_stage"]["gpus"],
        deterministic=False,
        precision=32,
        strategy="ddp_find_unused_parameters_true",  # GAN training requires this?
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 50,
        fast_dev_run=10 if debug else False,
        default_root_dir=log_dir,
        max_steps=conf["train_1st_stage"]["max_steps"],
        enable_progress_bar=debug,
        logger=False if debug else logger,
        enable_checkpointing=not debug,
        callbacks=callback_list,
    )
    trainer.fit(model, data)
    wandb.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
