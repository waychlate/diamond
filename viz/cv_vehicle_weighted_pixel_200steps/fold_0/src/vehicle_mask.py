# Copied from AnirudhhVenkat/kinemamba, src/mask_utils.py, commit 0c7566d.
# https://github.com/AnirudhhVenkat/kinemamba/blob/0c7566db7b2c4fb74004d03042f4ad752fc91f4e/src/mask_utils.py
import torch
import torch.nn.functional as F

"""Mask utilities for PIWM.

Design: soft mask emphasizing ego vehicle and traffic, with global (x-axis) and local (ego-centered) Gaussian attenuation.
"""

# Configurable parameters (set via set_mask_params)
MASK_W_BLUE: float = 1.0  # weight for traffic (blue)
MASK_W_GREEN: float = 0.6  # weight for ego (green)
MASK_SIGMA_GLOBAL_RATIO: float = 0.20  # global Gaussian sigma ratio relative to image width
MASK_SIGMA_EGO_SCALE: float = 0.50  # ego radial sigma scale relative to estimated size
MASK_SIGMA_MIN_PX: float = 2.0  # minimum sigma in pixels
MASK_DOWNSAMPLE_MODE: str = "bicubic"  # interpolation mode


def set_mask_params(w_blue: float, w_green: float, sigma_global_ratio: float, sigma_ego_scale: float, sigma_min_px: float, downsample_mode: str) -> None:
    """Set mask parameters (used by Denoiser from config)."""
    global MASK_W_BLUE, MASK_W_GREEN, MASK_SIGMA_GLOBAL_RATIO, MASK_SIGMA_EGO_SCALE, MASK_SIGMA_MIN_PX, MASK_DOWNSAMPLE_MODE
    MASK_W_BLUE = float(w_blue)
    MASK_W_GREEN = float(w_green)
    MASK_SIGMA_GLOBAL_RATIO = float(sigma_global_ratio)
    MASK_SIGMA_EGO_SCALE = float(sigma_ego_scale)
    MASK_SIGMA_MIN_PX = float(sigma_min_px)
    MASK_DOWNSAMPLE_MODE = str(downsample_mode)


def build_mask_from_fullres(frame_full: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    b, c, h, w = frame_full.shape 
    x = frame_full.add(1).div(2).clamp(0, 1)  
    r, g, bch = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    # Color thresholding
    max_rb = torch.maximum(r, bch)
    max_rg = torch.maximum(r, g)
    m_green = torch.logical_and(g > 0.35, (g - max_rb) > 0.12).float()
    m_blue = torch.logical_and(bch > 0.35, (bch - max_rg) > 0.12).float() 

    device, dtype = frame_full.device, frame_full.dtype

    # Ego centroid and variances computed in torch
    ys = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    mg_sum = m_green.sum(dim=(2, 3), keepdim=True)
    mg_valid = (mg_sum > 0)
    mg_sum_safe = mg_sum.clamp_min(1e-6)
    x_c = (m_green * xs).sum(dim=(2, 3), keepdim=True) / mg_sum_safe
    y_c = (m_green * ys).sum(dim=(2, 3), keepdim=True) / mg_sum_safe
    var_x = (m_green * (xs - x_c).pow(2)).sum(dim=(2, 3), keepdim=True) / mg_sum_safe
    var_y = (m_green * (ys - y_c).pow(2)).sum(dim=(2, 3), keepdim=True) / mg_sum_safe
    width_est = (12.0 * var_x).clamp_min(0).sqrt()
    height_est = (12.0 * var_y).clamp_min(0).sqrt()

    # hyper-params (configured via setter; keep signature unchanged)
    w_blue = MASK_W_BLUE
    w_green = MASK_W_GREEN
    sigma_global_ratio = MASK_SIGMA_GLOBAL_RATIO
    sigma_ego_scale = MASK_SIGMA_EGO_SCALE
    sigma_min_px = MASK_SIGMA_MIN_PX

    # Ego radial attenuation (elliptical Gaussian)
    sigma_ego_x = (sigma_ego_scale * width_est).clamp_min(sigma_min_px)
    sigma_ego_y = (sigma_ego_scale * height_est).clamp_min(sigma_min_px)
    ego_radial = torch.exp(-0.5 * (((xs - x_c) / sigma_ego_x).pow(2) + ((ys - y_c) / sigma_ego_y).pow(2)))
    m_ego_soft = m_green * ego_radial

    # Global horizontal Gaussian attenuation centered at ego
    sigma_global = max(float(sigma_global_ratio * w), 1.0)
    sigma_global_t = torch.tensor(sigma_global, device=device, dtype=dtype)
    w_global_x = torch.exp(-0.5 * ((xs - x_c) / sigma_global_t).pow(2))
    w_global_x = torch.where(mg_valid, w_global_x, torch.ones_like(w_global_x))

    # Merge masks
    m_soft = w_blue * m_blue + w_green * m_ego_soft
    m_soft = (m_soft * w_global_x).clamp(0, 1)

    # Downsample to low-res
    if MASK_DOWNSAMPLE_MODE in ("bilinear", "bicubic"):
        m_soft = F.interpolate(m_soft, size=(out_h, out_w), mode=MASK_DOWNSAMPLE_MODE, align_corners=False, antialias=True)
    else:
        m_soft = F.interpolate(m_soft, size=(out_h, out_w), mode=MASK_DOWNSAMPLE_MODE)
    return m_soft.clamp(0, 1)
