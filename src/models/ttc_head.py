from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TTCHeadConfig:
    in_channels: int = 64
    hidden_dim: int = 128
    pool_size: Tuple[int, int] = (4, 4)
    dropout: float = 0.1


class LatentTTCHead(nn.Module):
    """
    Predicts continuous Time-to-Collision (TTC) directly from DIAMOND's UNet bottleneck latents.
    Input shape: (B, C, H, W) or (B, T, C, H, W) where (C, H, W) is typically (64, 6, 40).
    Output shape: (B, 1) or (B, T, 1) representing continuous TTC (seconds or timesteps).
    """
    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 128,
        pool_size: Tuple[int, int] = (4, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size

        # Spatial feature reduction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, hidden_dim // 16), num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, hidden_dim // 16), num_channels=hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(pool_size),
        )

        flat_dim = hidden_dim * pool_size[0] * pool_size[1]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.SiLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        latents: Tensor,
        hx_cx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, None]:
        """
        latents: (B, C, H, W) or (B, T, C, H, W); each of the T latents is scored on its own.
        returns: ((B, 1) or (B, T, 1) continuous predicted TTC, None)
        hx_cx is unused and the second return value is always None; both only keep the old callers working.
        """
        if latents.ndim == 5:
            b, t, c, h, w = latents.shape
            x = self.conv(latents.reshape(b * t, c, h, w)).reshape(b, t, -1)
            return self.regressor(self.dropout(x)), None
        elif latents.ndim == 4:
            x = self.conv(latents).flatten(start_dim=1)
            return self.regressor(self.dropout(x)), None
        else:
            raise ValueError(f"Expected latents with 4 or 5 dimensions, got {latents.ndim}")
