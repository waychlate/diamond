"""Report parameter count, next-frame inference latency and peak memory for a DIAMOND checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cost(checkpoint, *, sigma_data=0.15, order=1, warmup=20, iters=200, device=None):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import Agent
    from models.diffusion import DiffusionSampler

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("agent", sd)
    # Read the architecture from the weights: conv_in stacks (context + noisy target) frames as channels.
    img_channels = 3
    num_cond = sd["denoiser.inner_model.conv_in.weight"].shape[1] // img_channels - 1
    cond_channels = sd["denoiser.inner_model.act_emb.0.weight"].shape[1] * num_cond
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=[
            f"agent.denoiser.inner_model.num_steps_conditioning={num_cond}",
            f"agent.denoiser.inner_model.cond_channels={cond_channels}",
            f"agent.denoiser.sigma_data={sigma_data}",
            f"world_model_env.diffusion_sampler.order={order}"])
    agent = Agent(instantiate(cfg.agent, num_actions=cfg.env.num_actions)).to(device).eval()
    agent.load(checkpoint)
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)

    params = {name: sum(p.numel() for p in getattr(agent, name).parameters())
              for name in ("denoiser", "rew_end_model", "actor_critic")}
    # KineMamba-style split of the denoiser: U-Net down path = "encoder", everything else
    # (action/noise conditioning, bottleneck, up path with skips, output conv) = "world_model".
    encoder = sum(p.numel() for n, p in agent.denoiser.inner_model.named_parameters()
                  if n.startswith(("conv_in", "unet.d_blocks", "unet.downsamples")))
    h, w = cfg.env.train.size
    obs = torch.rand(1, num_cond, img_channels, h, w, device=device) * 2 - 1
    act = torch.zeros(1, num_cond, dtype=torch.long, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            sampler.sample(obs, act)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            sampler.sample(obs, act)
            sync()
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {
        "checkpoint": str(checkpoint), "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "conditioning_frames": num_cond, "cond_channels": cond_channels, "image_size": [h, w],
        "denoising_steps": cfg.world_model_env.diffusion_sampler.num_steps_denoising, "sampler_order": order,
        "params": params, "params_total": sum(params.values()),
        "next_frame_params": params["denoiser"],  # Only the denoiser runs to predict pixels.
        "next_frame_params_mb_fp32": params["denoiser"] * 4 / 2**20,  # Same as KineMamba's check_mamba_size.py.
        "next_frame_params_by_part": {"encoder": encoder, "world_model": params["denoiser"] - encoder},
        "batch_size": 1, "warmup": warmup, "iters": iters,
        "latency_ms_mean": sum(times) / len(times), "latency_ms_median": times[len(times) // 2],
        "latency_ms_p95": times[int(len(times) * 0.95)],
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--order", type=int, default=1, help="1 = Euler, 2 = Heun")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = cost(args.checkpoint.resolve(), order=args.order, iters=args.iters, device=args.device)
    text = json.dumps(result, indent=2)
    # Same layout as KineMamba's check_sizes.py: part, param_count, size in MB (params * 4 bytes).
    from models.ttc_head import LatentTTCHead  # Untrained, no-LSTM head on the 64-channel bottleneck.
    parts = {**{k.title().replace("World_model", "World_Model"): v for k, v in result["next_frame_params_by_part"].items()},
             "TTC_Predictor": sum(p.numel() for p in LatentTTCHead(in_channels=64).parameters())}
    print("Part of pipeline, param_count, size in MB")
    for name, count in parts.items():
        print(f"{name}: {count}, {count * 4 / 1024**2}")
    print(f"Total MB: {sum(parts.values()) * 4 / 1024**2}")
    if args.output:
        args.output.write_text(text + "\n")
