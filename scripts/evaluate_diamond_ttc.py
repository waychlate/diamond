import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Add src to sys.path for DIAMOND imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add TTC_Prediction to sys.path for legacy TTC model imports if available
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "TTC_Prediction"))

from agent import Agent
from data import Dataset, BatchSampler, collate_segments_to_batch
from models.diffusion import DiffusionSampler
from models.ttc_head import LatentTTCHead

OmegaConf.register_new_resolver("eval", eval)


def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Computes Pearson and Spearman correlation coefficients and p-values safely.
    """
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
        }
    
    if SCIPY_AVAILABLE:
        p_res = stats.pearsonr(x, y)
        s_res = stats.spearmanr(x, y)
        return {
            "pearson_r": float(p_res.statistic) if hasattr(p_res, "statistic") else float(p_res[0]),
            "pearson_p": float(p_res.pvalue) if hasattr(p_res, "pvalue") else float(p_res[1]),
            "spearman_rho": float(s_res.statistic) if hasattr(s_res, "statistic") else float(s_res[0]),
            "spearman_p": float(s_res.pvalue) if hasattr(s_res, "pvalue") else float(s_res[1]),
        }
    else:
        p_r = float(np.corrcoef(x, y)[0, 1])
        return {
            "pearson_r": p_r,
            "pearson_p": 0.0,
            "spearman_rho": p_r,
            "spearman_p": 0.0,
        }


@torch.no_grad()
def evaluate_diamond_ttc(
    checkpoint_path: str,
    ttc_model_path: str,
    dataset_path: str,
    mode: str = "latent",
    num_episodes: int = 50,
    context_len: int = 20,
    rollout_steps: int = 30,
    dt: float = 0.1,
    max_ttc: Optional[float] = None,
    output_dir: str = "visualizations/latent_ttc_eval",
):
    """
    Evaluates Time-to-Collision (TTC) prediction performance with context_len history frames
    and evaluates step-by-step TTC loss and visual pixel reconstruction loss over 1+ lookahead steps into the future.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating DIAMOND TTC in [{mode.upper()}] mode on device: {device}")
    
    # 1. Load Hydra Config & DIAMOND Agent
    config_dir = "../config"
    with initialize(version_base="1.3", config_path=config_dir):
        cfg = compose(config_name="trainer", overrides=["env=highway"])
        
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    print("Instantiating DIAMOND Agent...")
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    
    if os.path.exists(checkpoint_path):
        print(f"Loading DIAMOND checkpoint from {checkpoint_path}...")
        agent.load(checkpoint_path)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")
        
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    print(f"DIAMOND conditioning history steps: {num_cond}")
    
    # 2. Load TTC Predictor Model
    ttc_predictor = None
    if mode == "latent":
        print(f"Loading Latent TTC Head from {ttc_model_path}...")
        in_channels = cfg.agent.denoiser.inner_model.channels[-1] if hasattr(cfg.agent.denoiser.inner_model, "channels") else 64
        use_lstm = True
        hidden_dim = 128
        
        if os.path.exists(ttc_model_path):
            ckpt = torch.load(ttc_model_path, map_location=device)
            if isinstance(ckpt, dict) and "config" in ckpt:
                use_lstm = ckpt["config"].get("use_temporal_lstm", True)
                hidden_dim = ckpt["config"].get("hidden_dim", 128)
                if "max_ttc" in ckpt and max_ttc is None:
                    max_ttc = ckpt["max_ttc"]
            
            ttc_predictor = LatentTTCHead(in_channels=in_channels, hidden_dim=hidden_dim, use_temporal_lstm=use_lstm).to(device).eval()
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            ttc_predictor.load_state_dict(sd)
            max_ttc_str = f"{max_ttc:.2f}s" if (max_ttc is not None and max_ttc > 0) else "uncapped"
            print(f"Latent TTC Head weights loaded successfully (use_lstm={use_lstm}, max_ttc={max_ttc_str}).")
        else:
            ttc_predictor = LatentTTCHead(in_channels=in_channels, hidden_dim=hidden_dim, use_temporal_lstm=use_lstm).to(device).eval()
            print(f"Warning: {ttc_model_path} not found. Proceeding with initial weights for evaluation structure.")
    else:
        try:
            from scripts.model import VideoTTCPredictor
            print(f"Loading Legacy Video TTC Predictor from {ttc_model_path}...")
            ttc_predictor = VideoTTCPredictor(hidden_dim=128, action_dim=16, use_actions=True, in_channels=9).to(device).eval()
            if os.path.exists(ttc_model_path):
                ckpt = torch.load(ttc_model_path, map_location=device)
                sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                ttc_predictor.load_state_dict(sd)
                print("Video TTC Predictor weights loaded.")
        except (ImportError, ModuleNotFoundError):
            print("Legacy VideoTTCPredictor not found, defaulting to latent mode.")

    # 3. Setup Dataset
    test_dataset_path = Path(dataset_path) / "test"
    if not test_dataset_path.exists():
        test_dataset_path = Path(dataset_path)
    print(f"Loading dataset from {test_dataset_path}...")
    test_dataset = Dataset(test_dataset_path, "test_dataset")
    test_dataset.load_from_default_path()
    
    seq_len = (num_cond + context_len - 1) + rollout_steps
    batch_sampler = BatchSampler(test_dataset, rank=0, world_size=1, batch_size=1, seq_length=seq_len, sample_weights=None)
    data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_segments_to_batch)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Rollout & Predict
    print(f"\nStarting DIAMOND Rollout & TTC Evaluation over {num_episodes} episodes with {context_len}-frame context...")
    data_iterator = iter(data_loader)
    
    step_maes = [[] for _ in range(rollout_steps)]
    step_mses = [[] for _ in range(rollout_steps)]
    step_pixel_mses = [[] for _ in range(rollout_steps)]
    step_pixel_maes = [[] for _ in range(rollout_steps)]
    step_pixel_psnrs = [[] for _ in range(rollout_steps)]
    
    all_records: List[Dict[str, Any]] = []
    all_gt: List[List[float]] = []
    all_pred: List[List[float]] = []
    all_ep_pixel_mses: List[List[float]] = []
    
    actual_evaluated_episodes = 0
    
    for ep_idx in range(num_episodes):
        try:
            batch = next(data_iterator)
        except StopIteration:
            print(f"Reached end of dataset after {ep_idx} episodes.")
            break
            
        actual_evaluated_episodes += 1
        obs = batch.obs.to(device)  # (1, seq_len, 3, H, W) in range [-1, 1]
        act = batch.act.to(device)  # (1, seq_len)
        end = batch.end.to(device)  # (1, seq_len)
        
        # Check ground truth crash index
        crash_indices = torch.where(end[0] == 1)[0]
        has_crash = len(crash_indices) > 0
        crash_idx = crash_indices[0].item() if has_crash else -1
        
        has_obs_ttc = (
            hasattr(batch, "info")
            and isinstance(batch.info, (list, tuple))
            and len(batch.info) > 0
            and isinstance(batch.info[0], dict)
            and "obs_ttc" in batch.info[0]
        )
        if ep_idx == 0:
            print(f"Ground truth obs_ttc detected in batch: {has_obs_ttc}")
            if has_obs_ttc:
                print(f"First 5 ground truth TTC values: {batch.info[0]['obs_ttc'][:5].tolist()}")

        # --- Phase 1: Ingest real history frames to warm up the LSTM memory ---
        hx_cx = None
        context_gt_ttc = []
        context_pred_ttc = []
        
        for t in range(context_len):
            window_obs = obs[:, t : t + num_cond]
            window_act = act[:, t : t + num_cond]
            curr_idx = num_cond + t - 1
            
            # Ground truth TTC for context step
            if has_obs_ttc:
                raw_val = float(batch.info[0]["obs_ttc"][curr_idx].item())
                true_ttc = max(raw_val, 0.0)
                if max_ttc is not None and max_ttc > 0:
                    true_ttc = min(true_ttc, max_ttc)
            elif hasattr(batch, "info") and isinstance(batch.info, dict) and "obs_ttc" in batch.info:
                raw_val = float(batch.info["obs_ttc"][0, curr_idx].item())
                true_ttc = max(raw_val, 0.0)
                if max_ttc is not None and max_ttc > 0:
                    true_ttc = min(true_ttc, max_ttc)
            elif has_crash and curr_idx <= crash_idx:
                true_ttc = (crash_idx - curr_idx) * dt
                if max_ttc is not None and max_ttc > 0:
                    true_ttc = min(true_ttc, max_ttc)
            else:
                true_ttc = max_ttc if (max_ttc is not None and max_ttc > 0) else 15.0
            context_gt_ttc.append(true_ttc)
            
            if mode == "latent" and ttc_predictor is not None:
                z = agent.denoiser.extract_latent(window_obs, window_act)
                pred_ttc, hx_cx = ttc_predictor(z.unsqueeze(1), hx_cx)
                context_pred_ttc.append(pred_ttc.squeeze().item())
            else:
                context_pred_ttc.append(0.0)

        curr_gt = context_gt_ttc[-1]
        curr_pred = context_pred_ttc[-1]
        curr_error = abs(curr_pred - curr_gt)
        if (ep_idx + 1) % 5 == 0 or ep_idx == 0 or ep_idx == num_episodes - 1:
            print(f"\nEpisode {ep_idx + 1}/{num_episodes}: Primed with {context_len} real frames.")
            print(f"  -> Current Moment (T=0) TTC: GT = {curr_gt:.2f}s | Pred = {curr_pred:.2f}s (Error: {curr_error:.2f}s)")
            print(f"  -> Rolling out {rollout_steps} lookahead steps into future dream...")

        # --- Phase 2: Autoregressive dream rollout into the future (+1 to +rollout_steps) ---
        history_obs = obs[:, context_len - 1 : context_len - 1 + num_cond].clone()
        history_act = act[:, context_len - 1 : context_len - 1 + num_cond].clone()
        
        future_gt_ttc = []
        future_pred_ttc = []
        ep_pixel_mses = []
        
        for step in range(rollout_steps):
            future_idx = (num_cond + context_len - 1) + step
            curr_act = act[:, future_idx : future_idx + 1] if future_idx < act.shape[1] else act[:, -1:]
            
            # 1. Rollout next observation in world model dream
            next_obs, _ = sampler.sample(history_obs, history_act)
            
            # 2. Compute visual pixel reconstruction loss against true next frame
            gt_obs = obs[:, future_idx] if future_idx < obs.shape[1] else obs[:, -1]
            pixel_mse = float(F.mse_loss(next_obs, gt_obs).item())
            pixel_mae = float(F.l1_loss(next_obs, gt_obs).item())
            # Dynamic range for image tensors in [-1, 1] is 2.0 -> MAX^2 = 4.0
            pixel_psnr = float(10.0 * np.log10(4.0 / max(pixel_mse, 1e-8)))
            
            # 3. Advance dream history buffer
            history_obs = torch.cat([history_obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
            history_act = torch.cat([history_act[:, 1:], curr_act], dim=1)
            
            # 4. Predict TTC from updated dream state using primed LSTM memory
            if mode == "latent" and ttc_predictor is not None:
                z = agent.denoiser.extract_latent(history_obs, history_act)
                pred_ttc, hx_cx = ttc_predictor(z.unsqueeze(1), hx_cx)
                pred_val = float(pred_ttc.squeeze().item())
            else:
                pred_val = 0.0
                
            # 5. Ground truth future TTC for step + 1
            if has_obs_ttc and future_idx < len(batch.info[0]["obs_ttc"]):
                raw_val = float(batch.info[0]["obs_ttc"][future_idx].item())
                true_ttc = max(raw_val, 0.0)
                if max_ttc is not None and max_ttc > 0:
                    true_ttc = min(true_ttc, max_ttc)
            elif has_crash and future_idx <= crash_idx:
                true_ttc = (crash_idx - future_idx) * dt
                if max_ttc is not None and max_ttc > 0:
                    true_ttc = min(true_ttc, max_ttc)
            else:
                true_ttc = max_ttc if (max_ttc is not None and max_ttc > 0) else 15.0
                
            future_gt_ttc.append(true_ttc)
            future_pred_ttc.append(pred_val)
            ep_pixel_mses.append(pixel_mse)
            
            error_mae = abs(pred_val - true_ttc)
            error_mse = (pred_val - true_ttc) ** 2
            step_maes[step].append(error_mae)
            step_mses[step].append(error_mse)
            step_pixel_mses[step].append(pixel_mse)
            step_pixel_maes[step].append(pixel_mae)
            step_pixel_psnrs[step].append(pixel_psnr)
            
            all_records.append({
                "episode": ep_idx + 1,
                "lookahead_step": step + 1,
                "ground_truth_ttc": round(true_ttc, 4),
                "predicted_ttc": round(pred_val, 4),
                "abs_error": round(error_mae, 4),
                "squared_error": round(error_mse, 4),
                "pixel_mse": round(pixel_mse, 6),
                "pixel_mae": round(pixel_mae, 6),
                "pixel_psnr": round(pixel_psnr, 4),
            })
            
        all_gt.append(future_gt_ttc)
        all_pred.append(future_pred_ttc)
        all_ep_pixel_mses.append(ep_pixel_mses)

        # Plot episode trajectory for first 15 episodes or if small episode count
        if ep_idx < 15 or ep_idx == num_episodes - 1:
            past_x = list(range(-context_len + 1, 1))
            future_x = list(range(1, rollout_steps + 1))

            plt.figure(figsize=(10, 4.5), dpi=200)
            plt.plot(past_x, context_gt_ttc, color="gray", linestyle="--", label="Ground Truth (Past Context)", linewidth=1.8)
            plt.plot(past_x, context_pred_ttc, color="darkcyan", linestyle="-", label="TTC Tracking (Past Context)", linewidth=2.0)
            plt.plot(future_x, future_gt_ttc, color="green", linestyle="--", label="Ground Truth (Future)", linewidth=2.0)
            plt.plot(future_x, future_pred_ttc, color="royalblue", linestyle="-", label=f"Predicted TTC ({mode.capitalize()} Dream)", linewidth=2.2)
            plt.axvline(x=0, color="crimson", linestyle=":", linewidth=2, label="Current Moment (T=0, Dream Begins)")
            
            plt.title(f"Episode {ep_idx + 1}: TTC Tracking with {context_len}-Frame Context & {rollout_steps}-Step Lookahead")
            plt.xlabel("Timeline Steps (Negative = Past Context, Positive = Future Lookahead)")
            plt.ylabel("TTC (seconds)")
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            ep_plot_path = os.path.join(output_dir, f"episode_{ep_idx + 1}_ttc_trajectory.png")
            plt.savefig(ep_plot_path)
            plt.close()

    # 1. Export Raw Step-by-Step Predictions CSV (Including Pixel Loss Metrics)
    csv_path = os.path.join(output_dir, "eval_ttc_predictions.csv")
    with open(csv_path, mode="w", newline="") as f:
        fieldnames = [
            "episode",
            "lookahead_step",
            "ground_truth_ttc",
            "predicted_ttc",
            "abs_error",
            "squared_error",
            "pixel_mse",
            "pixel_mae",
            "pixel_psnr"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved step-by-step predictions CSV to {csv_path}")

    # 2. Export Lookahead Horizon Metrics CSV
    horizon_csv_path = os.path.join(output_dir, "eval_horizon_metrics.csv")
    with open(horizon_csv_path, mode="w", newline="") as f:
        fieldnames = [
            "lookahead_step",
            "mean_mse",
            "std_mse",
            "sem_mse",
            "mean_mae",
            "std_mae",
            "mean_rmse",
            "mean_pixel_mse",
            "std_pixel_mse",
            "sem_pixel_mse",
            "mean_pixel_psnr"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(rollout_steps):
            errs = step_maes[step]
            sq_errs = step_mses[step]
            p_mses = step_pixel_mses[step]
            p_psnrs = step_pixel_psnrs[step]
            
            m_mse = float(np.mean(sq_errs)) if len(sq_errs) > 0 else 0.0
            s_mse = float(np.std(sq_errs)) if len(sq_errs) > 0 else 0.0
            sem_mse = s_mse / np.sqrt(len(sq_errs)) if len(sq_errs) > 0 else 0.0
            
            m_mae = float(np.mean(errs)) if len(errs) > 0 else 0.0
            s_mae = float(np.std(errs)) if len(errs) > 0 else 0.0
            m_rmse = float(np.sqrt(m_mse))
            
            m_pmse = float(np.mean(p_mses)) if len(p_mses) > 0 else 0.0
            s_pmse = float(np.std(p_mses)) if len(p_mses) > 0 else 0.0
            sem_pmse = s_pmse / np.sqrt(len(p_mses)) if len(p_mses) > 0 else 0.0
            m_psnr = float(np.mean(p_psnrs)) if len(p_psnrs) > 0 else 0.0
            
            writer.writerow({
                "lookahead_step": step + 1,
                "mean_mse": round(m_mse, 4),
                "std_mse": round(s_mse, 4),
                "sem_mse": round(sem_mse, 4),
                "mean_mae": round(m_mae, 4),
                "std_mae": round(s_mae, 4),
                "mean_rmse": round(m_rmse, 4),
                "mean_pixel_mse": round(m_pmse, 6),
                "std_pixel_mse": round(s_pmse, 6),
                "sem_pixel_mse": round(sem_pmse, 6),
                "mean_pixel_psnr": round(m_psnr, 4),
            })
    print(f"Saved lookahead horizon metrics CSV to {horizon_csv_path}")

    # 3. Specific Analysis: Zero and Lowest TTC MSE Instances
    sorted_records = sorted(all_records, key=lambda r: r["squared_error"])
    zero_mse_records = [r for r in sorted_records if r["squared_error"] <= 1e-4]
    top_50_lowest_records = sorted_records[: min(50, len(sorted_records))]
    
    zero_lowest_csv_path = os.path.join(output_dir, "eval_zero_lowest_ttc_instances.csv")
    with open(zero_lowest_csv_path, mode="w", newline="") as f:
        fieldnames = [
            "rank",
            "episode",
            "lookahead_step",
            "ground_truth_ttc",
            "predicted_ttc",
            "abs_error",
            "squared_error",
            "pixel_mse",
            "pixel_psnr"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(top_50_lowest_records, 1):
            row = {"rank": rank, **r}
            writer.writerow(row)
            
    zero_lowest_json_path = os.path.join(output_dir, "eval_zero_lowest_ttc_instances.json")
    with open(zero_lowest_json_path, "w") as f:
        json.dump({
            "num_exact_or_near_zero_mse": len(zero_mse_records),
            "top_50_lowest_instances": top_50_lowest_records
        }, f, indent=4)
    print(f"Saved zero/lowest TTC MSE instances ({len(zero_mse_records)} near-zero, top {len(top_50_lowest_records)} logged) to {zero_lowest_csv_path}")

    # 4. Correlation Analysis: Pixel Loss vs TTC Loss
    pix_mse_arr = np.array([r["pixel_mse"] for r in all_records])
    pix_mae_arr = np.array([r["pixel_mae"] for r in all_records])
    pix_psnr_arr = np.array([r["pixel_psnr"] for r in all_records])
    ttc_mse_arr = np.array([r["squared_error"] for r in all_records])
    ttc_mae_arr = np.array([r["abs_error"] for r in all_records])
    
    corr_pix_mse_vs_ttc_mse = compute_correlations(pix_mse_arr, ttc_mse_arr)
    corr_pix_mse_vs_ttc_mae = compute_correlations(pix_mse_arr, ttc_mae_arr)
    corr_pix_psnr_vs_ttc_mse = compute_correlations(pix_psnr_arr, ttc_mse_arr)
    corr_pix_psnr_vs_ttc_mae = compute_correlations(pix_psnr_arr, ttc_mae_arr)
    
    # Step-by-step correlation breakdown
    step_correlations = []
    for s in range(rollout_steps):
        s_pix_mse = np.array(step_pixel_mses[s])
        s_ttc_mse = np.array(step_mses[s])
        s_corr = compute_correlations(s_pix_mse, s_ttc_mse)
        step_correlations.append({
            "lookahead_step": s + 1,
            "pearson_r": round(s_corr["pearson_r"], 4),
            "pearson_p": round(s_corr["pearson_p"], 6),
            "spearman_rho": round(s_corr["spearman_rho"], 4),
            "spearman_p": round(s_corr["spearman_p"], 6),
            "mean_pixel_mse": round(float(np.mean(s_pix_mse)), 6) if len(s_pix_mse) > 0 else 0.0,
            "mean_ttc_mse": round(float(np.mean(s_ttc_mse)), 4) if len(s_ttc_mse) > 0 else 0.0,
        })
        
    # Episode-level correlation breakdown
    episode_correlations = []
    for ep_i in range(len(all_gt)):
        ep_recs = [r for r in all_records if r["episode"] == ep_i + 1]
        if len(ep_recs) > 1:
            ep_pmse = np.array([r["pixel_mse"] for r in ep_recs])
            ep_tmse = np.array([r["squared_error"] for r in ep_recs])
            ep_corr = compute_correlations(ep_pmse, ep_tmse)
            episode_correlations.append({
                "episode": ep_i + 1,
                "pearson_r": round(ep_corr["pearson_r"], 4),
                "spearman_rho": round(ep_corr["spearman_rho"], 4),
                "mean_pixel_mse": round(float(np.mean(ep_pmse)), 6),
                "mean_ttc_mse": round(float(np.mean(ep_tmse)), 4),
            })

    correlation_summary = {
        "overall_pixel_mse_vs_ttc_mse": corr_pix_mse_vs_ttc_mse,
        "overall_pixel_mse_vs_ttc_mae": corr_pix_mse_vs_ttc_mae,
        "overall_pixel_psnr_vs_ttc_mse": corr_pix_psnr_vs_ttc_mse,
        "overall_pixel_psnr_vs_ttc_mae": corr_pix_psnr_vs_ttc_mae,
        "step_by_step_correlations": step_correlations,
        "episode_correlations": episode_correlations,
        "total_data_points": len(all_records),
    }
    
    corr_json_path = os.path.join(output_dir, "eval_pixel_vs_ttc_correlation.json")
    with open(corr_json_path, "w") as f:
        json.dump(correlation_summary, f, indent=4)
    print(f"Saved pixel vs TTC correlation summary to {corr_json_path}")

    # 5. Export Compressed NumPy NPZ Dataset
    npz_path = os.path.join(output_dir, "eval_data.npz")
    np.savez_compressed(
        npz_path,
        ground_truth=np.array(all_gt),
        predictions=np.array(all_pred),
        step_maes=np.array(step_maes),
        step_mses=np.array(step_mses),
        pixel_mses=np.array(step_pixel_mses),
        pixel_maes=np.array(step_pixel_maes),
        pixel_psnrs=np.array(step_pixel_psnrs)
    )
    print(f"Saved compressed NumPy arrays to {npz_path}")

    # 6. Compute Overall Summary Statistics and Save JSON
    if len(all_records) > 0:
        overall_mae = float(np.mean([r["abs_error"] for r in all_records]))
        overall_mse = float(np.mean([r["squared_error"] for r in all_records]))
        overall_rmse = float(np.sqrt(overall_mse))
        overall_pixel_mse = float(np.mean(pix_mse_arr))
        overall_pixel_psnr = float(np.mean(pix_psnr_arr))
        
        gt_flat = np.array([r["ground_truth_ttc"] for r in all_records])
        pred_flat = np.array([r["predicted_ttc"] for r in all_records])
        corr_ttc = float(np.corrcoef(gt_flat, pred_flat)[0, 1]) if len(gt_flat) > 1 and np.std(gt_flat) > 0 and np.std(pred_flat) > 0 else 0.0

        # R^2 calculation
        ss_res = float(np.sum((gt_flat - pred_flat) ** 2))
        ss_tot = float(np.sum((gt_flat - np.mean(gt_flat)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        summary = {
            "num_episodes": len(all_gt),
            "context_len": context_len,
            "rollout_steps": rollout_steps,
            "total_evaluated_points": len(all_records),
            "overall_mae_seconds": round(overall_mae, 4),
            "overall_mse_seconds2": round(overall_mse, 4),
            "overall_rmse_seconds": round(overall_rmse, 4),
            "r2_score": round(r2, 4),
            "pearson_correlation_r": round(corr_ttc, 4),
            "overall_pixel_mse": round(overall_pixel_mse, 6),
            "overall_pixel_psnr_db": round(overall_pixel_psnr, 4),
            "pixel_mse_vs_ttc_mse_pearson_r": round(corr_pix_mse_vs_ttc_mse["pearson_r"], 4),
            "pixel_mse_vs_ttc_mse_spearman_rho": round(corr_pix_mse_vs_ttc_mse["spearman_rho"], 4),
            "num_zero_or_near_zero_mse_instances": len(zero_mse_records),
            "lowest_recorded_mse": round(top_50_lowest_records[0]["squared_error"], 6) if top_50_lowest_records else 0.0,
            "mode": mode
        }
        json_path = os.path.join(output_dir, "eval_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved evaluation summary JSON to {json_path}")
        print(f"\n=================== EVALUATION SUMMARY ===================")
        print(f"Evaluated Episodes:         {len(all_gt)}")
        print(f"Context Length:             {context_len} frames")
        print(f"Lookahead Horizon:          +{rollout_steps} steps")
        print(f"Total Evaluated Points:     {len(all_records)}")
        print(f"Overall TTC MAE:            {overall_mae:.4f} s")
        print(f"Overall TTC MSE:            {overall_mse:.4f} s²")
        print(f"Overall TTC RMSE:           {overall_rmse:.4f} s")
        print(f"TTC Pearson Correlation R:  {corr_ttc:.4f}")
        print(f"Overall Dream Pixel MSE:    {overall_pixel_mse:.6f}")
        print(f"Overall Dream Pixel PSNR:   {overall_pixel_psnr:.2f} dB")
        print(f"Pixel MSE vs TTC MSE r:     {corr_pix_mse_vs_ttc_mse['pearson_r']:.4f} (Spearman rho: {corr_pix_mse_vs_ttc_mse['spearman_rho']:.4f})")
        print(f"Zero / Near-Zero MSE Count: {len(zero_mse_records)}")
        if top_50_lowest_records:
            print(f"Lowest Recorded TTC MSE:    {top_50_lowest_records[0]['squared_error']:.6f} s² (Ep {top_50_lowest_records[0]['episode']}, Step +{top_50_lowest_records[0]['lookahead_step']})")
        print(f"==========================================================\n")

    # 7. Visualization 1: Lookahead Horizon Curve (Average MSE vs Lookahead Step)
    if any(len(errs) > 0 for errs in step_mses):
        lookahead_steps = np.arange(1, rollout_steps + 1)
        mean_mses = np.array([np.mean(errs) if len(errs) > 0 else 0.0 for errs in step_mses])
        std_mses = np.array([np.std(errs) if len(errs) > 0 else 0.0 for errs in step_mses])
        num_eps = max(1, len(all_gt))
        sem_mses = std_mses / np.sqrt(num_eps)

        plt.figure(figsize=(10, 5.5), dpi=300)
        for ep_i in range(len(all_gt)):
            ep_sq_errors = [(all_pred[ep_i][s] - all_gt[ep_i][s]) ** 2 for s in range(rollout_steps)]
            plt.plot(lookahead_steps, ep_sq_errors, color="slategray", alpha=0.18, linewidth=1.0,
                     label="Individual Episode MSE" if ep_i == 0 else "")
            
        plt.plot(lookahead_steps, mean_mses, marker="o", color="#d62728", linewidth=2.8,
                 label=f"Average MSE ({num_eps} Episodes)")
        plt.fill_between(lookahead_steps, np.maximum(0, mean_mses - sem_mses), mean_mses + sem_mses,
                         color="#d62728", alpha=0.2, label="±1 SEM")

        plt.title(f"TTC Prediction: Average MSE vs. Lookahead Horizon\n({context_len}-Frame Context Priming, Averaged over {num_eps} Episodes)",
                  fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Lookahead Step (+1, +2, ... +N steps into future dream)", fontsize=11)
        plt.ylabel(r"TTC Mean Squared Error $\mathrm{MSE}\ (\mathrm{s}^2)$", fontsize=11)
        plt.xticks(lookahead_steps[::2] if len(lookahead_steps) > 15 else lookahead_steps)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", fontsize=10, framealpha=0.9)
        plt.tight_layout()
        
        mse_plot_path = os.path.join(output_dir, "avg_mse_over_lookahead.png")
        plt.savefig(mse_plot_path)
        plt.savefig(os.path.join(output_dir, "latent_ttc_mse_horizon.png"))
        plt.close()
        print(f"Saved Average MSE Lookahead Plot to {mse_plot_path}")

    # 8. Visualization 2: Correlation Scatter Plot (Pixel Loss vs TTC Loss)
    if len(all_records) > 0:
        plt.figure(figsize=(9, 6), dpi=300)
        steps_col = np.array([r["lookahead_step"] for r in all_records])
        scatter = plt.scatter(
            pix_mse_arr,
            ttc_mse_arr,
            c=steps_col,
            cmap="viridis",
            alpha=0.65,
            edgecolors="none",
            s=35
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Lookahead Step (+1 to +30)", fontsize=10)
        
        # Fit linear regression line
        if np.std(pix_mse_arr) > 1e-8:
            slope, intercept = np.polyfit(pix_mse_arr, ttc_mse_arr, 1)
            x_line = np.linspace(np.min(pix_mse_arr), np.max(pix_mse_arr), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color="crimson", linestyle="--", linewidth=2.2,
                     label=f"Linear Fit: y = {slope:.2f}x + {intercept:.2f}")

        pr_r = corr_pix_mse_vs_ttc_mse["pearson_r"]
        sp_rho = corr_pix_mse_vs_ttc_mse["spearman_rho"]
        plt.title(f"Correlation: Visual Dream Pixel Error vs. TTC Prediction Error\n(Pearson r = {pr_r:.3f}, Spearman ρ = {sp_rho:.3f}, N = {len(all_records)})",
                  fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Visual Reconstruction Loss (Pixel MSE)", fontsize=11)
        plt.ylabel(r"TTC Prediction Error ($\mathrm{MSE}\ \mathrm{s}^2$)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        
        scatter_plot_path = os.path.join(output_dir, "pixel_vs_ttc_scatter.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        print(f"Saved Pixel vs TTC Scatter Plot to {scatter_plot_path}")

    # 9. Visualization 3: Dual-Axis Progression (Pixel Drift & TTC Error vs Lookahead)
    if any(len(errs) > 0 for errs in step_mses):
        lookahead_steps = np.arange(1, rollout_steps + 1)
        mean_mses = np.array([np.mean(errs) for errs in step_mses])
        mean_pmse = np.array([np.mean(p_errs) for p_errs in step_pixel_mses])
        
        fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
        
        color_ttc = "#d62728"
        ax1.set_xlabel("Lookahead Horizon (Steps into Future Dream)", fontsize=11)
        ax1.set_ylabel(r"TTC Prediction $\mathrm{MSE}\ (\mathrm{s}^2)$", color=color_ttc, fontsize=11)
        line1 = ax1.plot(lookahead_steps, mean_mses, color=color_ttc, marker="o", linewidth=2.5, label="TTC MSE")
        ax1.tick_params(axis="y", labelcolor=color_ttc)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        ax2 = ax1.twinx()
        color_pix = "#1f77b4"
        ax2.set_ylabel("Dream Reconstruction Pixel MSE", color=color_pix, fontsize=11)
        line2 = ax2.plot(lookahead_steps, mean_pmse, color=color_pix, marker="s", linestyle="-.", linewidth=2.5, label="Pixel MSE")
        ax2.tick_params(axis="y", labelcolor=color_pix)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
        
        plt.title(f"Dream Quality Degradation vs TTC Prediction Drift\n({len(all_gt)} Episodes Average)",
                  fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        
        dual_plot_path = os.path.join(output_dir, "pixel_vs_ttc_horizon.png")
        plt.savefig(dual_plot_path)
        plt.close()
        print(f"Saved Dual-Axis Horizon Plot to {dual_plot_path}")

    # 10. Visualization 4: Multi-Metric Horizon Dashboard
    if any(len(errs) > 0 for errs in step_mses):
        avg_maes = np.array([np.mean(errs) for errs in step_maes])
        avg_rmses = np.sqrt(mean_mses)
        avg_pmse = np.array([np.mean(p_errs) for p_errs in step_pixel_mses])
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=250)
        
        # Subplot 1: TTC MSE
        axes[0].plot(lookahead_steps, mean_mses, marker="o", color="#d62728", linewidth=2.2)
        axes[0].fill_between(lookahead_steps, np.maximum(0, mean_mses - sem_mses), mean_mses + sem_mses, color="#d62728", alpha=0.18)
        axes[0].set_title(r"TTC MSE ($\mathrm{s}^2$)", fontweight="bold")
        axes[0].set_xlabel("Lookahead Step")
        axes[0].set_ylabel(r"$\mathrm{MSE}\ (\mathrm{s}^2)$")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        # Subplot 2: TTC RMSE
        axes[1].plot(lookahead_steps, avg_rmses, marker="s", color="#ff7f0e", linewidth=2.2)
        axes[1].set_title("TTC RMSE (s)", fontweight="bold")
        axes[1].set_xlabel("Lookahead Step")
        axes[1].set_ylabel("RMSE (seconds)")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        # Subplot 3: TTC MAE
        axes[2].plot(lookahead_steps, avg_maes, marker="^", color="#2ca02c", linewidth=2.2)
        axes[2].set_title("TTC MAE (s)", fontweight="bold")
        axes[2].set_xlabel("Lookahead Step")
        axes[2].set_ylabel("MAE (seconds)")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        
        # Subplot 4: Dream Pixel MSE
        axes[3].plot(lookahead_steps, avg_pmse, marker="d", color="#1f77b4", linewidth=2.2)
        axes[3].set_title("Dream Pixel MSE", fontweight="bold")
        axes[3].set_xlabel("Lookahead Step")
        axes[3].set_ylabel("Pixel MSE")
        axes[3].grid(True, linestyle="--", alpha=0.5)
        
        fig.suptitle(f"Latent TTC Lookahead Horizon Dashboard ({len(all_gt)} Episodes Avg, {context_len}-Frame Context)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        multi_metric_path = os.path.join(output_dir, "latent_ttc_horizon_metrics.png")
        plt.savefig(multi_metric_path)
        plt.close()
        print(f"Saved Horizon Metrics Dashboard to {multi_metric_path}")

    print("\nDIAMOND TTC Evaluation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DIAMOND World Model Latent TTC Predictor")
    parser.add_argument("--checkpoint", type=str, default="diamond_highway_mcts.pt", help="Path to DIAMOND checkpoint")
    parser.add_argument("--ttc-model", type=str, default="checkpoints/best_latent_ttc.pt", help="Path to trained Latent TTC weights")
    parser.add_argument("--dataset_path", type=str, default="dataset_mcts", help="Path to dataset directory")
    parser.add_argument("--mode", type=str, default="latent", choices=["latent", "pixel"], help="Evaluation mode (latent or pixel)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to evaluate (default: 50)")
    parser.add_argument("--context_len", type=int, default=20, help="Number of past context frames to prime the LSTM (default: 20)")
    parser.add_argument("--rollout_steps", type=int, default=30, help="Number of future steps to rollout using DIAMOND")
    parser.add_argument("--dt", type=float, default=0.1, help="Time delta per step in seconds")
    parser.add_argument("--max_ttc", type=float, default=None, help="Fallback max TTC in seconds (default: None for uncapped)")
    parser.add_argument("--output_dir", type=str, default="visualizations/latent_ttc_eval", help="Directory to save output plots and metrics")
    
    args = parser.parse_args()
    evaluate_diamond_ttc(
        checkpoint_path=args.checkpoint,
        ttc_model_path=args.ttc_model,
        dataset_path=args.dataset_path,
        mode=args.mode,
        num_episodes=args.episodes,
        context_len=args.context_len,
        rollout_steps=args.rollout_steps,
        dt=args.dt,
        max_ttc=args.max_ttc,
        output_dir=args.output_dir
    )
