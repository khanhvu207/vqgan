import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py#L331
class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps

        torch.manual_seed(42)
        weight = torch.randn(num_tokens, codebook_dim)
        weight.data.uniform_(-math.sqrt(3.0 / num_tokens), math.sqrt(3.0 / num_tokens))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)

        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )

        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAQuantize(nn.Module):
    def __init__(self, code_dim, num_codes, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost

        self.embedding = EmbeddingEMA(num_codes, code_dim, decay=decay)

    def forward(self, z):
        B, C, H, W = z.shape
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = rearrange(z, "b h w c -> (b h w) c")

        dist = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * einsum(
                z_flattened,
                rearrange(self.embedding.weight, "n d -> d n"),
                "b d, d n -> b n",
            )
        )

        min_encoding_indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        encodings = F.one_hot(min_encoding_indices, self.num_codes).type(z.dtype)
        avg_use = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_use * torch.log(avg_use + 1e-10)))

        if self.training is True:
            encodings_sum = encodings.sum(dim=0)
            self.embedding.cluster_size_ema_update(encodings_sum)

            embedding_sum = einsum(
                rearrange(encodings, "n d -> d n"), z_flattened, "d n, n b -> d b"
            )
            self.embedding.embed_avg_ema_update(embedding_sum)
            self.embedding.weight_update(self.num_codes)

        loss = (
            self.commitment_cost * ((z_q.detach() - z) ** 2).mean()
        )  # + ((z_q - z.detach())**2).mean()
        z_q = z + (z_q - z).detach()  # Straight-through estimator
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        min_encoding_indices = rearrange(
            min_encoding_indices, "(b h w) -> b h w", **{"b": B, "h": H, "w": W}
        )

        return z_q, loss, (min_encoding_indices, perplexity)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        z_q = z_q.view(shape)
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        return z_q


class STQuantize(nn.Module):
    def __init__(self, code_dim, num_codes, commitment_cost=0.25):
        super().__init__()
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.kaiming_uniform_(self.embedding.weight.data)

    def forward(self, z):
        B, C, H, W = z.shape
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = rearrange(z, "b h w c -> (b h w) c")

        dist = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * einsum(
                z_flattened,
                rearrange(self.embedding.weight, "n d -> d n"),
                "b d, d n -> b n",
            )
        )

        min_encoding_indices = torch.argmin(dist, dim=1)

        encodings = F.one_hot(min_encoding_indices, self.num_codes).type(z.dtype)
        avg_use = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_use * torch.log(avg_use + 1e-10)))

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = (
            self.commitment_cost * ((z_q.detach() - z) ** 2).mean()
            + ((z_q - z.detach()) ** 2).mean()
        )

        z_q = (
            z + (z_q - z).detach()
        )  # Straight-through estimator, must be done after calculating the loss!

        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        min_encoding_indices = rearrange(
            min_encoding_indices, "(b h w) -> b h w", **{"b": B, "h": H, "w": W}
        )
        return z_q, loss, (min_encoding_indices, perplexity)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        z_q = z_q.view(shape)
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        return z_q
