from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor


class LatentTTCHead(nn.Module):
    """
    KineMamba's TTCPredictor (src/models/ttc_predictor.py at 391ac50) on DIAMOND's U-Net bottleneck.
    KineMamba feeds its head the 1024-d Mamba state. Here the 64×6×40 bottleneck is average-pooled
    (no parameters) to 64×2×8 = 1024 features, so the MLP is identical: 1024→256→128→1, ReLU,
    dropout 0.2, 295,425 parameters. With a dense bottleneck (in_features = its width) the head reads that
    vector directly, with no pooling.
    Input: (B, C, H, W), (B, T, C, H, W), (B, D) or (B, T, D). Output: (B, 1) or (B, T, 1), TTC in seconds.
    """
    def __init__(self, in_channels: int = 64, hidden_dim: int = 256, dropout: float = 0.2,
                 pool_size: Tuple[int, int] = (2, 8), in_features: Optional[int] = None) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.head = nn.Sequential(
            nn.Linear(in_features or in_channels * pool_size[0] * pool_size[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, latents: Tensor, hx_cx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, None]:
        """hx_cx is unused and the second return value is always None; both only keep the old callers working."""
        if latents.ndim in (2, 3):
            return self.head(latents), None
        if latents.ndim not in (4, 5):
            raise ValueError(f"Expected latents with 2 to 5 dimensions, got {latents.ndim}")
        lead, (c, h, w) = latents.shape[:-3], latents.shape[-3:]
        x = self.pool(latents.reshape(-1, c, h, w)).flatten(start_dim=1)
        return self.head(x).reshape(*lead, 1), None
