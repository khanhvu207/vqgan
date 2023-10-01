import gc
import math
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from autoencoder import VQModel
from modules.transformers import GPT, CfgNode, TokenDataModule
from utils import *


class GenerativeModel(pl.LightningModule):
    def __init__(self, stage_one_model, log_dir, **conf):
        super().__init__()
        self.save_hyperparameters(ignore="stage_one_model")
        self.conf = conf
        self.stage_one_model = stage_one_model
        self.log_dir = log_dir
        self.train_conf = self.conf["train_2nd_stage"]
        model_config = CfgNode(**self.conf["transformer"])
        self.transformer = GPT(model_config)
        self.sos_token = 0
        self.sample_shape = None
        self.sample_device = None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        B, L = x.shape
        c = torch.ones(B, 1) * self.sos_token
        c = c.long().to(x.device)

        x = x + 1  # Shift the indices by 1
        cx = torch.cat((c, x), dim=1)

        targets = x
        logits, _ = self.transformer(cx[:, :-1])
        logits = logits[:, c.shape[1] - 1 :]
        return logits, targets

    def training_step(self, batch, batch_idx):
        logits, targets = self(batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train/loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, targets = self(batch)

        if self.sample_shape is None:
            self.sample_shape = targets.shape
            self.sample_device = targets.device

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("val/loss", loss.item(), prog_bar=True, sync_dist=True)
        return loss

    @torch.inference_mode()
    def sample(self, batch_size, cond, length, top_k=50, top_p=None):
        cond = cond.repeat(batch_size, cond.shape[0])
        samples = self.transformer.generate(
            idx=cond,
            max_new_tokens=length,
            temperature=1.0,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
        )
        samples = samples[:, cond.shape[1] :]
        samples = samples - 1  # Map indices back to [0..codebook_size-1]
        return samples

    @torch.inference_mode()
    def decode_to_img(self, indices):
        quantizer = self.stage_one_model.quantize
        b = indices.shape[0]
        h = int(math.sqrt(indices.shape[1]))
        w = h
        ch = self.conf["vector_quantization"]["params"]["code_dim"]
        num_codes = self.conf["vector_quantization"]["params"]["num_codes"]
        assert (
            torch.any(indices > num_codes - 1).item() == False
        ), "Some indices is larger than the codebook size!"

        z_q = quantizer.get_codebook_entry(indices, shape=(b, h, w, ch))
        img = self.stage_one_model.decode(z_q)
        return img

    @torch.inference_mode()
    def on_validation_end(self):
        if self.trainer.is_global_zero is False:
            return

        c = torch.tensor([self.sos_token], dtype=torch.long, device=self.sample_device)
        generated_indices = self.sample(
            batch_size=16, cond=c, length=self.sample_shape[1]
        )

        generated_images = self.decode_to_img(generated_indices)
        generated_images = make_grid(outmap(generated_images.cpu()), nrow=8)

        # Visualize synthetic images
        fig, ax = plt.subplots(1, 1, dpi=150)
        ax.imshow(
            np.transpose(generated_images.numpy(), (1, 2, 0)), interpolation="nearest"
        )
        for axis in fig.axes:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        plt.tight_layout()

        img_name = f"{self.current_epoch}-generation.png"
        img_save_path = os.path.join(self.log_dir, img_name)
        plt.savefig(img_save_path)
        plt.clf()

        if self.logger is not None:
            self.logger.experiment.log(
                {
                    "visualizations": [
                        wandb.Image(
                            os.path.join(self.log_dir, img_name),
                            caption=f"Generation, Step {self.global_step}",
                        ),
                    ]
                },
                step=self.global_step,
            )
        plt.close()

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.train_conf["weight_decay"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.train_conf["lr"], betas=(0.9, 0.95)
        )
        return optimizer

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
        optimizer = self.optimizers()
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
    config_name = os.path.splitext(args["config"])[0].split("/")[1]
    log_dir = f"./logs/{config_name}/{run_id}"

    train_tokens, val_tokens = torch.load(os.path.join(log_dir, "image_tokens.pt"))
    print("train_tokens shape:", train_tokens.shape)
    print("val_tokens shape:", val_tokens.shape)
    data = TokenDataModule(train_tokens, val_tokens, **conf["train_2nd_stage"])

    stage_one_model = VQModel.load_from_checkpoint(
        os.path.join(log_dir, run_id + ".ckpt")
    )
    for p in stage_one_model.parameters():
        p.requires_grad_(False)
    model = GenerativeModel(stage_one_model, log_dir, **conf)
    print(model)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=log_dir,
        filename="generator_weights",
        save_top_k=1,
        mode="min",
    )
    callback_list = [checkpoint_callback] if not debug else []

    logger = WandbLogger(
        project="generative",
        entity="kvu207",
        name=f"[2]{run_id}",
        save_dir=log_dir,
        offline=debug,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=conf["train_2nd_stage"]["gpus"],
        deterministic=False,
        precision=32,  # Use fp32 for now
        # strategy="ddp_find_unused_parameters_true",
        strategy="auto",
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 50,
        fast_dev_run=5 if debug else False,
        default_root_dir=log_dir,
        val_check_interval=0.25,
        max_steps=conf["train_2nd_stage"]["max_steps"],
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
