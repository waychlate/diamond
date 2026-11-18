import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, KineMambaUNet, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None
    # > 0 adds a dense bottleneck of this width (see UNet); needs img_size to size the flattened map.
    dense_bottleneck_dim: int = 0
    img_size: Optional[List[int]] = None
    # True uses KineMambaUNet (see blocks.py): noise conditions every ResBlock, actions enter only at the
    # dense bottleneck (needs dense_bottleneck_dim > 0), KineMamba's additive skips, 4 downsamples.
    kinemamba_unet: bool = False


KINEMAMBA_ACT_DIM = 32  # KineMamba's per-step action embedding (Mamba2LatentDynamics)


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        act_dim = KINEMAMBA_ACT_DIM if cfg.kinemamba_unet else cfg.cond_channels // cfg.num_steps_conditioning
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, act_dim),
            nn.Flatten(),  # b t e -> b (t e)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        features = 0
        if cfg.dense_bottleneck_dim > 0:
            if cfg.img_size is None:
                raise ValueError("dense_bottleneck_dim > 0 needs img_size")
            scale = 2 ** (len(cfg.channels) - (0 if cfg.kinemamba_unet else 1))
            features = cfg.channels[-1] * math.ceil(cfg.img_size[0] / scale) * math.ceil(cfg.img_size[1] / scale)
        if cfg.kinemamba_unet:
            self.unet = KineMambaUNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths,
                                      cfg.dense_bottleneck_dim, features, KINEMAMBA_ACT_DIM * cfg.num_steps_conditioning)
        else:
            self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths,
                             cfg.dense_bottleneck_dim, features)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def extract_latent(self, obs: Tensor, act: Tensor) -> Tensor:
        if obs.ndim == 5:
            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)
        else:
            b, _, h, w = obs.shape

        dummy_c_noise = torch.zeros(b, device=obs.device, dtype=obs.dtype)
        dummy_next = torch.zeros(b, self.cfg.img_channels, h, w, device=obs.device, dtype=obs.dtype)
        x = self.conv_in(torch.cat((obs, dummy_next), dim=1))

        # The TTC head reads the dense hidden vector when there is one (as KineMamba's reads h), else the map.
        latent, _, hidden = self.unet.encode(x, *self._conditioning(dummy_c_noise, act))
        return latent if hidden is None else hidden

    def _conditioning(self, c_noise: Tensor, act: Tensor):
        """UNet args after x: (cond,) with actions in every block, or (cond, actions) with actions at the bottleneck."""
        if self.cfg.kinemamba_unet:
            return self.cond_proj(self.noise_emb(c_noise)), self.act_emb(act)
        return (self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act)),)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, *self._conditioning(c_noise, act))
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x

