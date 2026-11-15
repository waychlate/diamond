from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import LossAndLogs
from ..ttc_head import LatentTTCHead
from vehicle_mask import build_mask_from_fullres


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    pixel_mask_gain: float = 0.0
    pixel_mae_weight: float = 0.0
    # 0 with pixel_mae_weight > 0 trains on the vehicle-weighted pixel MAE alone (KineMamba's pixel loss).
    diffusion_mse_weight: float = 1.0
    # > 0 adds a TTC head on the U-Net bottleneck and trains it jointly: weight * L1 of the next frame's TTC (seconds).
    # The gradient reaches conv_in, the down blocks, the bottleneck and the conditioning layers, not the up path.
    ttc_loss_weight: float = 0.0
    ttc_hidden_dim: int = 256
    ttc_dropout: float = 0.2


def vehicle_weighted_pixel_mae(prediction: Tensor, target: Tensor, gain: float) -> Tensor:
    """Full-image MAE with additional target-derived vehicle weights, shared over RGB."""
    if target.numel() == 0:
        return prediction.sum() * 0.0
    with torch.no_grad():
        mask = build_mask_from_fullres(target, target.shape[-2], target.shape[-1])
        weights = 1.0 + gain * mask
    return ((prediction - target).abs() * weights).sum() / (weights.sum() * target.shape[1])


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        for name in ("pixel_mask_gain", "pixel_mae_weight", "diffusion_mse_weight", "ttc_loss_weight"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if cfg.pixel_mae_weight == 0 and cfg.diffusion_mse_weight == 0:
            raise ValueError("pixel_mae_weight and diffusion_mse_weight cannot both be zero")
        self.inner_model = InnerModel(cfg.inner_model)
        self.ttc_head = None
        if cfg.ttc_loss_weight > 0:
            self.ttc_head = LatentTTCHead(in_channels=cfg.inner_model.channels[-1], hidden_dim=cfg.ttc_hidden_dim,
                                          dropout=cfg.ttc_dropout,
                                          in_features=cfg.inner_model.dense_bottleneck_dim or None)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape 
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)
    
    def extract_latent(self, obs: Tensor, act: Tensor) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        return self.inner_model.extract_latent(rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, batch: Batch) -> LossAndLogs:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n
        if seq_length <= 0:
            raise ValueError("A denoising batch must contain at least one frame after the conditioning context")

        all_obs = batch.obs.clone()
        ttc = None
        if self.ttc_head is not None:
            if not all("obs_ttc" in info for info in batch.info):
                raise ValueError("ttc_loss_weight > 0 needs obs_ttc in the dataset")
            ttc = torch.stack([info["obs_ttc"] for info in batch.info]).to(self.device).float().clamp(min=0.0)
        loss = next(self.inner_model.parameters()).sum() * 0.0
        diffusion_mse = 0
        pixel_mae = 0
        ttc_l1 = 0
        valid_steps = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = batch.act[:, i : n + i]
            mask = batch.mask_padding[:, n + i]
            if not mask.any():
                continue
            valid_steps += 1

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma)
            model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)

            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
            # The original preconditioned diffusion objective; always logged, weighted into the loss.
            mse = F.mse_loss(model_output[mask], target[mask])
            diffusion_mse = diffusion_mse + mse.detach()
            if self.cfg.diffusion_mse_weight > 0:
                loss = loss + self.cfg.diffusion_mse_weight * mse
            if self.cfg.pixel_mae_weight > 0:
                # Use the continuous reconstruction: wrap_model_output quantizes and
                # detaches it, which would prevent a pixel loss from teaching the model.
                reconstructed = cs.c_skip * noisy_next_obs + cs.c_out * model_output
                weighted_mae = vehicle_weighted_pixel_mae(
                    reconstructed[mask], batch.obs[:, n + i][mask], self.cfg.pixel_mask_gain
                )
                pixel_mae = pixel_mae + weighted_mae
                loss = loss + self.cfg.pixel_mae_weight * weighted_mae

            if ttc is not None:
                # A clean pass (no noise, zero next frame) like the one the head reads at inference.
                pred_ttc = self.ttc_head(self.extract_latent(all_obs[:, i : n + i], act))[0].squeeze(-1)
                step_ttc = F.l1_loss(pred_ttc[mask], ttc[:, n + i][mask])
                ttc_l1 = ttc_l1 + step_ttc
                loss = loss + self.cfg.ttc_loss_weight * step_ttc

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        # Padding contributes neither error nor averaging weight. An entirely padded
        # batch still returns a differentiable zero for the training loop.
        denominator = max(valid_steps, 1)
        loss = loss / denominator
        logs = {"loss_denoising": loss.detach(),
                "loss_diffusion_mse": torch.as_tensor(diffusion_mse / denominator, device=loss.device).detach()}
        if self.cfg.pixel_mae_weight > 0:
            logs["loss_weighted_pixel_mae"] = torch.as_tensor(pixel_mae / denominator, device=loss.device).detach()
        if ttc is not None:
            logs["loss_ttc"] = torch.as_tensor(ttc_l1 / denominator, device=loss.device).detach()
        return loss, logs
