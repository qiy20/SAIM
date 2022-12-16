# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from util.blocks import Block, GaussianConv2d


class SaimViT(nn.Module):
    """
    Pretrain vision transformer backbone with SAIM
    """

    def __init__(self,
                 # vision transformer backbone
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 # decoder
                 query_depth=12, share_weight=False,
                 prediction_head_type='MLP',
                 # loss function
                 gaussian_kernel_size=None, gaussian_sigma=None,
                 loss_type='L2', norm_pix_loss=True):
        super().__init__()

        # patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # encoder
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])

        # decoder
        if share_weight:
            self.query_blocks = self.blocks
        else:
            self.query_blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(query_depth)])
        self.depth = depth
        self.step = depth // query_depth

        # prediction head
        self.norm = norm_layer(embed_dim)
        if prediction_head_type == 'LINEAR':
            self.prediction_head = nn.Linear(embed_dim, patch_size ** 2 * 3)
        elif prediction_head_type == 'MLP':
            self.prediction_head = Mlp(embed_dim, int(embed_dim * mlp_ratio), patch_size ** 2 * 3)

        # define loss parameters
        self.loss_type = loss_type
        self.norm_pix_loss = norm_pix_loss
        if gaussian_kernel_size is not None and gaussian_sigma is not None:
            self.gaussian_blur = GaussianConv2d(3, gaussian_kernel_size, gaussian_sigma)
        else:
            self.gaussian_blur = nn.Identity()

        # initialize weight
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def generate_attention_mask(self, x):
        """
        Generate permutation mask(content mask and query mask)
       """
        N, L, D = x.shape  # batch, length, dim
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # content mask
        full_mask = torch.full((N, L, L), -math.inf, device=x.device)
        no_mask = torch.zeros((N, L, L), device=x.device)
        mask_h = torch.where(noise.unsqueeze(-1) < noise.unsqueeze(1), full_mask, no_mask)  # broadcast-->N*L*L

        # query mask
        mask_g = torch.where(noise.unsqueeze(-1) <= noise.unsqueeze(1), full_mask, no_mask)

        # consider cls_token
        top_padding = torch.full((N, 1, L), -math.inf, device=x.device)  # cls token can't see other tokens
        left_padding = torch.zeros((N, L + 1, 1), device=x.device)  # other tokens can see cls token
        mask_h = torch.cat((top_padding, mask_h), dim=1)
        mask_h = torch.cat((left_padding, mask_h), dim=2)
        mask_g = torch.cat((top_padding, mask_g), dim=1)
        mask_g = torch.cat((left_padding, mask_g), dim=2)
        return mask_h.unsqueeze(1), mask_g.unsqueeze(1)

    def forward_saim(self, x):
        # embed patches
        x = self.patch_embed(x)
        mask_h, mask_g = self.generate_attention_mask(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # permutation mask
        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)  # use fixed pos-embedding, not learnable tensor
        for i in range(self.depth):
            h = self.blocks[i](h, mask=mask_h)
            if (i + 1) % self.step == 0:
                g = self.query_blocks[i // self.step](g, h, mask=mask_g)
        g = self.norm(g)
        g = self.prediction_head(g)

        return g

    def forward_loss(self, imgs, pred):
        imgs = self.gaussian_blur(imgs)
        target = self.patchify(imgs)
        pred = pred[:, 1:, :]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        if self.loss_type == 'L1':
            loss = (pred - target).abs()
        elif self.loss_type == 'L2':
            loss = (pred - target) ** 2
        return loss.mean()

    def forward(self, imgs):
        pred = self.forward_saim(imgs)
        loss = self.forward_loss(imgs, pred)
        return loss


def saim_base(**kwargs):
    return SaimViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(2022)
    model = saim_base(img_size=224,  norm_pix_loss=False,
                      prediction_head_type='MLP', loss_type='L2',
                      query_depth=12, share_weight=False,
                      gaussian_kernel_size=9, gaussian_sigma=1)
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    print(model(x))
