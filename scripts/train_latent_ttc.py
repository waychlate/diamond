import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import Agent
from data import Dataset, BatchSampler, collate_segments_to_batch, Batch
from models.ttc_head import LatentTTCHead

OmegaConf.register_new_resolver("eval", eval)


def compute_ttc_targets_for_batch(
    batch: Batch,
    num_cond: int,
    dt: float = 0.1,
    max_ttc: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes ground truth TTC targets for each step in a batch after conditioning.
    Uses continuous obs_ttc from batch.info if present, otherwise falls back to step-counting.
    If max_ttc is None or <= 0, targets are left uncapped.
    """
    b, seq_len = batch.end.shape
    num_targets = seq_len - num_cond
    device = batch.end.device
    
    fallback_val = max_ttc if (max_ttc is not None and max_ttc > 0) else 15.0
    targets = torch.full((b, num_targets, 1), fallback_val, dtype=torch.float32, device=device)
    valid_masks = batch.mask_padding[:, num_cond - 1 : num_cond - 1 + num_targets].clone()
    
    has_obs_ttc_list = (
        hasattr(batch, "info")
        and isinstance(batch.info, (list, tuple))
        and len(batch.info) > 0
        and isinstance(batch.info[0], dict)
        and "obs_ttc" in batch.info[0]
    )
    has_obs_ttc_dict = hasattr(batch, "info") and isinstance(batch.info, dict) and "obs_ttc" in batch.info
    
    if has_obs_ttc_list:
        raw_ttc = torch.stack([s_info["obs_ttc"] for s_info in batch.info]).to(device).float()  # (B, seq_len)
        for step in range(num_targets):
            curr_idx = num_cond + step - 1
            val = torch.clamp(raw_ttc[:, curr_idx], min=0.0)
            if max_ttc is not None and max_ttc > 0:
                val = torch.clamp(val, max=max_ttc)
            targets[:, step, 0] = val
    elif has_obs_ttc_dict:
        raw_ttc = batch.info["obs_ttc"].to(device).float()  # (B, seq_len)
        for step in range(num_targets):
            curr_idx = num_cond + step - 1
            val = torch.clamp(raw_ttc[:, curr_idx], min=0.0)
            if max_ttc is not None and max_ttc > 0:
                val = torch.clamp(val, max=max_ttc)
            targets[:, step, 0] = val
    else:
        # Fallback: step-counting until crash
        for i in range(b):
            end_seq = batch.end[i]
            crash_indices = torch.where(end_seq == 1)[0]
            has_crash = len(crash_indices) > 0
            crash_idx = crash_indices[0].item() if has_crash else -1
            
            for step in range(num_targets):
                curr_idx = num_cond + step - 1
                if has_crash and curr_idx <= crash_idx:
                    ttc_seconds = (crash_idx - curr_idx) * dt
                    targets[i, step, 0] = min(ttc_seconds, fallback_val)
                else:
                    targets[i, step, 0] = fallback_val

    return targets, valid_masks


def extract_latents_for_sequence(
    agent: Agent,
    obs: torch.Tensor,
    act: torch.Tensor,
    num_cond: int,
    context_len: int
) -> torch.Tensor:
    """
    Extracts latents across all time steps in a sequence in a single batched pass.
    obs: (B, seq_len, 3, H, W)
    act: (B, seq_len)
    Returns: (B, context_len, C, H_mid, W_mid)
    """
    b, seq_len, c, h, w = obs.shape
    windows_obs = torch.stack([obs[:, t : t + num_cond] for t in range(context_len)], dim=1)
    windows_act = torch.stack([act[:, t : t + num_cond] for t in range(context_len)], dim=1)
    
    flat_obs = windows_obs.reshape(b * context_len, num_cond, c, h, w)
    flat_act = windows_act.reshape(b * context_len, num_cond)
    
    flat_z = agent.denoiser.extract_latent(flat_obs, flat_act)
    _, c_mid, h_mid, w_mid = flat_z.shape
    return flat_z.reshape(b, context_len, c_mid, h_mid, w_mid)


def calculate_dataset_max_ttc(dataset: Dataset, dt: float = 0.1, fallback_max: float = 5.0) -> float:
    """
    Scans the dataset to determine the maximum collision horizon observed.
    """
    print("Analyzing dataset to determine maximum TTC horizon...")
    max_ttc_observed = 0.0
    total_valid = 0
    
    for ep_id in range(min(dataset.num_episodes, 500)):
        try:
            ep = dataset.load_episode(ep_id)
            if hasattr(ep, "info") and isinstance(ep.info, dict) and "obs_ttc" in ep.info:
                ttc_vals = ep.info["obs_ttc"]
                valid_vals = ttc_vals[(ttc_vals > 0) & (ttc_vals < 100.0)]
                if len(valid_vals) > 0:
                    max_ttc_observed = max(max_ttc_observed, float(valid_vals.max().item()))
                    total_valid += 1
            else:
                crashes = torch.where(ep.end == 1)[0]
                if len(crashes) > 0:
                    total_valid += 1
                    max_ttc_observed = max(max_ttc_observed, float(crashes[0].item() * dt))
        except Exception:
            continue
            
    if total_valid > 0 and max_ttc_observed > 0:
        max_ttc = float(max_ttc_observed)
        print(f"Computed dataset max TTC: {max_ttc:.2f}s ({total_valid} episodes inspected).")
        return max_ttc
    else:
        print(f"Using default max TTC: {fallback_max:.2f}s")
        return fallback_max


def train_latent_ttc(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Training Latent TTC Predictor on device: {device}")
    
    # 1. Load DIAMOND Config & Backbone Agent
    config_dir = "../config"
    with initialize(version_base="1.3", config_path=config_dir):
        cfg = compose(config_name="trainer", overrides=["env=highway"])
        
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    print("Instantiating DIAMOND Agent...")
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    
    if os.path.exists(args.checkpoint):
        print(f"Loading DIAMOND checkpoint from {args.checkpoint}...")
        agent.load(args.checkpoint, load_denoiser=True, load_rew_end_model=False, load_actor_critic=False)
        print("DIAMOND Denoiser weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with initialized weights.")
        
    # Freeze Denoiser weights
    for param in agent.denoiser.parameters():
        param.requires_grad = False
    agent.denoiser.eval()
    
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    print(f"DIAMOND conditioning history steps: {num_cond}")
    
    # 2. Setup Dataset
    train_path = Path(args.dataset_path) / "train"
    if not train_path.exists():
        train_path = Path(args.dataset_path)
        
    test_path = Path(args.dataset_path) / "test"
    if not test_path.exists():
        test_path = train_path
        
    print(f"Loading train dataset from {train_path}...")
    train_dataset = Dataset(train_path, "train_dataset")
    train_dataset.load_from_default_path()
    
    print(f"Loading val dataset from {test_path}...")
    val_dataset = Dataset(test_path, "test_dataset")
    val_dataset.load_from_default_path()
    
    print(f"Loaded train dataset: {train_dataset.num_episodes} episodes ({train_dataset.num_steps} steps).")
    print(f"Loaded val dataset: {val_dataset.num_episodes} episodes ({val_dataset.num_steps} steps).")
    
    if train_dataset.num_episodes == 0:
        raise RuntimeError(
            f"No episodes found in train dataset ({train_path})! "
            f"Please verify 'info.pt' exists in {train_path} or run scripts/convert_and_process.py first."
        )
    
    # Determine max TTC cap from dataset
    if args.max_ttc is not None and args.max_ttc > 0:
        max_ttc = args.max_ttc
        print(f"Using user-specified max TTC cap: {max_ttc:.2f}s")
    else:
        max_ttc = None
        print("Training with UNCAPPED continuous TTC targets.")
        
    seq_len = num_cond + args.context_len
    train_sampler = BatchSampler(train_dataset, rank=0, world_size=1, batch_size=args.batch_size, seq_length=seq_len, sample_weights=None)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_segments_to_batch, num_workers=args.num_workers)
    
    val_sampler = BatchSampler(val_dataset, rank=0, world_size=1, batch_size=args.batch_size, seq_length=seq_len, sample_weights=None)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_segments_to_batch, num_workers=args.num_workers)
    
    # 3. Instantiate LatentTTCHead
    in_channels = cfg.agent.denoiser.inner_model.channels[-1] if hasattr(cfg.agent.denoiser.inner_model, "channels") else 64
    ttc_head = LatentTTCHead(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        use_temporal_lstm=not args.no_lstm,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(ttc_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val_mae = float("inf")
    start_epoch = 1

    # Checkpoint Auto-Resume
    if args.resume and os.path.exists(args.save_path):
        print(f"Loading checkpoint to resume from {args.save_path}...")
        try:
            ckpt = torch.load(args.save_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ttc_head.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                best_val_mae = ckpt.get("val_mae", float("inf"))
                start_epoch = ckpt.get("epoch", 0) + 1
                for _ in range(start_epoch - 1):
                    scheduler.step()
                print(f"--> Resumed from epoch {start_epoch}/{args.epochs} (Best Val MAE so far: {best_val_mae:.4f}s)")
        except Exception as e:
            print(f"Warning: Failed to resume checkpoint from {args.save_path} ({e}). Starting fresh.")
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    val_steps = max(10, args.steps_per_epoch // 5)

    print(f"\nStarting training from epoch {start_epoch} to {args.epochs} ({args.steps_per_epoch} steps/epoch) with context_len={args.context_len} frames...")
    for epoch in range(start_epoch, args.epochs + 1):
        ttc_head.train()
        train_loss_total = 0.0
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for _ in pbar:
            batch = next(train_iter)
            obs = batch.obs.to(device)  # (B, seq_len, 3, H, W)
            act = batch.act.to(device)  # (B, seq_len)
            
            # Ground truth targets
            targets, masks = compute_ttc_targets_for_batch(batch, num_cond=num_cond, dt=args.dt, max_ttc=max_ttc)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Fast vectorized latent extraction across the 20-frame context window
            with torch.no_grad():
                latents_seq = extract_latents_for_sequence(agent, obs, act, num_cond, args.context_len)
            
            # Forward pass through LatentTTCHead
            pred_ttc, _ = ttc_head(latents_seq)  # (B, context_len, 1)
            
            # Compute loss only on valid non-padded steps
            loss = F.smooth_l1_loss(pred_ttc[masks], targets[masks])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ttc_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss_total / max(1, args.steps_per_epoch)
        
        # --- Validation Loop ---
        ttc_head.eval()
        val_mae_total = 0.0
        val_mse_total = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for _ in range(val_steps):
                batch = next(val_iter)
                obs = batch.obs.to(device)
                act = batch.act.to(device)
                
                targets, masks = compute_ttc_targets_for_batch(batch, num_cond=num_cond, dt=args.dt, max_ttc=max_ttc)
                targets = targets.to(device)
                masks = masks.to(device)
                
                latents_seq = extract_latents_for_sequence(agent, obs, act, num_cond, args.context_len)
                pred_ttc, _ = ttc_head(latents_seq)
                
                valid_preds = pred_ttc[masks]
                valid_targs = targets[masks]
                
                if len(valid_preds) > 0:
                    mae = F.l1_loss(valid_preds, valid_targs, reduction="sum").item()
                    mse = F.mse_loss(valid_preds, valid_targs, reduction="sum").item()
                    val_mae_total += mae
                    val_mse_total += mse
                    val_samples += len(valid_preds)
                    
        avg_val_mae = val_mae_total / max(1, val_samples)
        avg_val_mse = val_mse_total / max(1, val_samples)
        
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f}s | Val RMSE: {avg_val_mse**0.5:.4f}s")
        
        # Save Best Checkpoint
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": ttc_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": avg_val_mae,
                "max_ttc": max_ttc,
                "config": {
                    "in_channels": in_channels,
                    "hidden_dim": args.hidden_dim,
                    "use_temporal_lstm": not args.no_lstm,
                    "context_len": args.context_len,
                    "dt": args.dt,
                }
            }
            torch.save(checkpoint_payload, args.save_path)
            print(f"--> Saved new best model with Val MAE: {best_val_mae:.4f}s to {args.save_path}")

    print(f"\nTraining Complete. Best Validation MAE: {best_val_mae:.4f}s. Saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Latent TTC Head directly from DIAMOND's UNet features")
    parser.add_argument("--checkpoint", type=str, default="diamond_highway_mcts.pt", help="Path to pretrained DIAMOND model")
    parser.add_argument("--dataset_path", type=str, default="dataset_mcts", help="Path to processed DIAMOND dataset")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_latent_ttc.pt", help="Output path for trained TTC weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Number of batches/steps per epoch (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for LatentTTCHead")
    parser.add_argument("--context_len", type=int, default=20, help="Number of history frames / steps observed (default: 20)")
    parser.add_argument("--no_lstm", action="store_true", help="Disable temporal LSTM (defaults to using LSTM for 20 frames)")
    parser.add_argument("--dt", type=float, default=0.1, help="Delta time per step in seconds (default: 0.1s for 10Hz)")
    parser.add_argument("--max_ttc", type=float, default=None, help="Max TTC cap in seconds (default: None for uncapped)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers (default: 0 for single-process memory safety)")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from existing save_path checkpoint if found (default: True)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    train_latent_ttc(args)
