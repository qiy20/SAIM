import math

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None, return_attention=False):
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
        The universal attention block can be used as both self-attention and cross-attention.
        q,k,v can define separately.
        If we only assign a value to q, it's a self-attention block;
        if we assign values for q and k, it's a cross-attention block.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        x = q + self.drop_path(self.attn(self.norm1(q), self.norm1(k), self.norm1(v), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GaussianConv2d(nn.Module):
    def __init__(self, channels=3, kernel_size=9, sigma=1):
        super().__init__()
        position = torch.stack(torch.meshgrid([torch.arange(kernel_size), torch.arange(kernel_size)]), dim=-1)
        mean = torch.tensor([(kernel_size - 1) // 2, (kernel_size - 1) // 2])
        std = torch.tensor([sigma, sigma])
        kernel = 1 / (2 * math.pi * torch.prod(std, dim=-1)) * math.e ** (-((position - mean) ** 2 / std ** 2).sum(-1)/2)
        kernel = kernel / kernel.sum()

        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, input):
        return F.conv2d(input, weight=self.weight, groups=self.groups, padding=self.padding)