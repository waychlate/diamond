"""Exhaustive one-step pixel MAE evaluation, independent of TTC heads."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import Dataset
from pixel_mask import build_mask_from_fullres


def error_totals(prediction, target, mask):
    """Return sufficient statistics; an empty mask contributes no masked observations."""
    error = (prediction - target).abs().double()
    return {
        "absolute_error": error.sum().item(),
        "pixel_channels": error.numel(),
        "masked_absolute_error": (error * mask.double()).sum().item(),
        "mask_weight_channels": mask.double().sum().item() * error.shape[1],
        "transitions": error.shape[0],
        "empty_mask_transitions": int((mask.flatten(1).sum(1) == 0).sum().item()),
    }


def add_totals(total, part):
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def with_mae(totals):
    return {**totals,
            "unmasked_mae": totals["absolute_error"] / totals["pixel_channels"],
            "masked_mae": (totals["masked_absolute_error"] / totals["mask_weight_channels"]
                           if totals["mask_weight_channels"] else None)}


def transition_batches(episode, num_cond, batch_size):
    observations = episode.obs
    if "final_observation" in episode.info:
        observations = torch.cat((observations, episode.info["final_observation"].unsqueeze(0)))
    # A terminal action can predict its final frame; context cannot cross a termination.
    dead = torch.nonzero(episode.dead, as_tuple=False).flatten()
    stop = min(len(observations), int(dead[0]) + 2) if len(dead) else len(observations)
    indices = list(range(num_cond, min(stop, len(episode.act) + 1)))
    for offset in range(0, len(indices), batch_size):
        targets = indices[offset:offset + batch_size]
        yield (targets,
               torch.stack([observations[t - num_cond:t] for t in targets]),
               torch.stack([episode.act[t - num_cond:t] for t in targets]),
               torch.stack([observations[t] for t in targets]))


def save_example(path, context, target, prediction, mask):
    def rgb(tensor):
        return tensor.detach().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
    target_rgb = rgb(target)
    weight = mask.detach().cpu().squeeze(0).numpy()[..., None]
    mask_rgb = np.repeat((weight * 255).astype(np.uint8), 3, axis=2)
    overlay = (target_rgb * (1 - 0.5 * weight) + np.array([255, 0, 0]) * 0.5 * weight).astype(np.uint8)
    # Left to right: last real context, target, prediction, grayscale mask, overlay.
    Image.fromarray(np.concatenate([rgb(context), target_rgb, rgb(prediction), mask_rgb, overlay], axis=1)).save(path)


@torch.inference_mode()
def evaluate(dataset_path, checkpoint, output, *, config_path=None, sigma_data=0.15,
             checkpoint_training_sigma=0.5, seed=0, batch_size=32, order=1, examples=6,
             device=None):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import Agent
    from models.diffusion import DiffusionSampler

    checkpoint, dataset_path, output = Path(checkpoint).resolve(), Path(dataset_path).resolve(), Path(output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (dataset_path / "info.pt").is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_path / 'info.pt'}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    if config_path:
        cfg = OmegaConf.load(config_path)
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
            cfg = compose(config_name="trainer")
    cfg.agent.denoiser.sigma_data = sigma_data
    cfg.world_model_env.diffusion_sampler.order = order
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, output / "config.yaml")
    agent = Agent(instantiate(cfg.agent, num_actions=cfg.env.num_actions)).to(device).eval()
    agent.load(checkpoint, load_rew_end_model=False, load_actor_critic=False)
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)
    dataset = Dataset(dataset_path)
    dataset.load_from_default_path()
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    totals, episode_rows, examples_saved = {}, [], 0
    with (output / "transitions.csv").open("w", newline="") as handle:
        fields = ["episode_id", "target_index", "unmasked_mae", "masked_mae", "mask_weight_channels"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode_id in tqdm(range(dataset.num_episodes), desc="Pixel MAE episodes"):
            episode_totals = {}
            for indices, history_obs, history_act, target in transition_batches(dataset.load_episode(episode_id), num_cond, batch_size):
                history_obs, history_act, target = history_obs.to(device), history_act.to(device), target.to(device)
                prediction, _ = sampler.sample(history_obs, history_act)
                mask = build_mask_from_fullres(target, target.shape[-2], target.shape[-1])
                if not torch.isfinite(prediction).all():
                    raise ValueError(f"Non-finite prediction in episode {episode_id}")
                for i, target_index in enumerate(indices):
                    part = error_totals(prediction[i:i+1], target[i:i+1], mask[i:i+1])
                    add_totals(episode_totals, part)
                    row = with_mae(part)
                    writer.writerow({"episode_id": episode_id, "target_index": target_index,
                                     **{key: row[key] for key in fields[2:]}})
                if examples_saved < examples:
                    save_example(output / f"example_{examples_saved:02d}.png", history_obs[0, -1], target[0], prediction[0], mask[0])
                    examples_saved += 1
            if episode_totals:
                add_totals(totals, episode_totals)
                episode_rows.append({"episode_id": episode_id, **with_mae(episode_totals)})
    if not totals:
        raise ValueError("No eligible transitions; episodes are shorter than the conditioning context")
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    summary = {
        "protocol": "one_step_real_context", "value_range": [-1, 1],
        "aggregation": "sum_absolute_errors_divided_by_sum_pixel_or_mask_weights",
        "mask": "KineMamba 0c7566d soft vehicle mask; blue=1, green=0.6; target-derived",
        "checkpoint": str(checkpoint), "checkpoint_training_sigma_data": checkpoint_training_sigma,
        "evaluation_sigma_data": sigma_data, "sigma_data_override": sigma_data != checkpoint_training_sigma,
        "dataset": str(dataset_path), "seed": seed, "batch_size": batch_size,
        "sampler_order": order, "conditioning_frames": num_cond,
        "episodes": len(episode_rows), "dataset_episodes": dataset.num_episodes,
        "skipped_short_episodes": dataset.num_episodes - len(episode_rows),
        **with_mae(totals),
    }
    for metric in ("masked_mae", "unmasked_mae"):
        values = [row[metric] for row in episode_rows if row[metric] is not None]
        summary[metric + "_episode_mean"] = float(np.mean(values)) if values else None
        summary[metric + "_episode_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    # Array tasks use the summary as a completion marker; publish it atomically.
    pending = output / "summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pending.replace(output / "summary.json")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Converted split directory containing info.pt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sigma-data", type=float, default=0.15)
    parser.add_argument("--checkpoint-training-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    evaluate(args.dataset, args.checkpoint, args.output, config_path=args.config,
             sigma_data=args.sigma_data, checkpoint_training_sigma=args.checkpoint_training_sigma,
             seed=args.seed, batch_size=args.batch_size, order=args.order,
             examples=args.examples, device=args.device)
