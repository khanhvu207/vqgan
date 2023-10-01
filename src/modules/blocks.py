import math
from collections import OrderedDict
from functools import partial

import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


@attr.s(eq=False)
class Conv2d(nn.Module):  # TODO: simplify to standard PyTorch Conv2d
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
    kw: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        w = torch.empty((self.n_out, self.n_in, self.kw, self.kw), dtype=torch.float32)
        w.data.normal_(std=1 / math.sqrt(self.n_in * self.kw**2))

        b = torch.zeros((self.n_out,), dtype=torch.float32)

        self.weight, self.bias = nn.Parameter(w), nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, padding=(self.kw - 1) // 2)


@attr.s(eq=False, repr=False)
class EncoderBlock(nn.Module):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers**2)

        make_conv = partial(Conv2d)
        self.id_path = (
            make_conv(self.n_in, self.n_out, 1)
            if self.n_in != self.n_out
            else nn.Identity()
        )
        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", make_conv(self.n_in, self.n_hid, 3)),
                    ("relu_2", nn.ReLU()),
                    ("conv_2", make_conv(self.n_hid, self.n_hid, 3)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", make_conv(self.n_hid, self.n_hid, 3)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", make_conv(self.n_hid, self.n_out, 1)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
class OpenAIEncoder(nn.Module):
    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        group_count = 4
        blk_range = range(self.n_blk_per_group)
        n_layers = group_count * self.n_blk_per_group
        make_conv = partial(Conv2d)
        make_blk = partial(EncoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("input", make_conv(self.input_channels, 1 * self.n_hid, 5)),
                    (
                        "group_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(1 * self.n_hid, 1 * self.n_hid),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                1 * self.n_hid
                                                if i == 0
                                                else 2 * self.n_hid,
                                                2 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                2 * self.n_hid
                                                if i == 0
                                                else 4 * self.n_hid,
                                                4 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                4 * self.n_hid
                                                if i == 0
                                                else 8 * self.n_hid,
                                                8 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                ]
                            )
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", nn.ReLU()),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

        self.output_channels = 8 * self.n_hid
        self.output_stide = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"input has {x.shape[1]} channels but model built for {self.input_channels}"
            )
        if x.dtype != torch.float32:
            raise ValueError("input must have dtype torch.float32")

        return self.blocks(x)


@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers**2)

        make_conv = partial(Conv2d)
        self.id_path = (
            make_conv(self.n_in, self.n_out, 1)
            if self.n_in != self.n_out
            else nn.Identity()
        )
        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", make_conv(self.n_in, self.n_hid, 1)),
                    ("relu_2", nn.ReLU()),
                    ("conv_2", make_conv(self.n_hid, self.n_hid, 3)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", make_conv(self.n_hid, self.n_hid, 3)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", make_conv(self.n_hid, self.n_out, 3)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
class OpenAIDecoder(nn.Module):
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        group_count = 4
        blk_range = range(self.n_blk_per_group)
        n_layers = group_count * self.n_blk_per_group
        make_conv = partial(Conv2d)
        make_blk = partial(DecoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    (
                        "group_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                self.n_init
                                                if i == 0
                                                else 8 * self.n_hid,
                                                8 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                8 * self.n_hid
                                                if i == 0
                                                else 4 * self.n_hid,
                                                4 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                4 * self.n_hid
                                                if i == 0
                                                else 2 * self.n_hid,
                                                2 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                2 * self.n_hid
                                                if i == 0
                                                else 1 * self.n_hid,
                                                1 * self.n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                ]
                            )
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", nn.ReLU()),
                                    (
                                        "conv",
                                        make_conv(
                                            1 * self.n_hid, self.output_channels, 1
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.dtype != torch.float32:
            raise ValueError("input must have dtype torch.float32")

        return self.blocks(x)
