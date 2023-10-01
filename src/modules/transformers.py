"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import json
import math
import os
import random
import sys
from ast import literal_eval

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """monotonous bookkeeping"""
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """a lightweight configuration class inspired by yacs"""

    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """need to have a helper to support nested indentation for pretty printing"""
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """return a dict representation of the config"""
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == "--"
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(
                obj, leaf_key
            ), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


# -----------------------------------------------------------------------------


class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        self.indices = np.arange(len(self.tokens))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.tokens[self.indices[idx]]
        s = s.view(-1)
        return s


class TokenDataModule(pl.LightningDataModule):
    def __init__(self, train_tokens, val_tokens, **conf):
        super().__init__()
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.batch_size = conf["batch_size"]
        self.num_workers = conf["num_workers"]
        self.num_gpus = len(conf["gpus"])
        self.name = "ffhq-2nd_stage"

    def setup(self, stage=None):
        self.train_data = TokenDataset(
            self.train_tokens,
        )
        self.val_data = TokenDataset(
            self.val_tokens,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size // self.num_gpus,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_loader()


# -----------------------------------------------------------------------------
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict(
                {
                    # names follow the huggingface naming conventions
                    # GPT-1
                    "openai-gpt": dict(
                        n_layer=12, n_head=12, n_embd=768
                    ),  # 117M params
                    # GPT-2 configs
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    "gpt2-medium": dict(
                        n_layer=24, n_head=16, n_embd=1024
                    ),  # 350M params
                    "gpt2-large": dict(
                        n_layer=36, n_head=20, n_embd=1280
                    ),  # 774M params
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                    # Gophers
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    # (there are a number more...)
                    # I made these tiny models up
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
                }[config.model_type]
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        top_p=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs >= top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = torch.zeros_like(
                    probs, dtype=sorted_indices_to_remove.dtype
                ).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                # Normalize the probabilities again
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
