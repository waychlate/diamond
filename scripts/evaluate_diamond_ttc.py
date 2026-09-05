import os
import sys
import csv
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

# Add src to sys.path for DIAMOND imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add TTC_Prediction to sys.path for legacy TTC model imports if available
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "TTC_Prediction"))

from agent import Agent
from data import Dataset, BatchSampler, collate_segments_to_batch
from models.diffusion import DiffusionSampler
from models.ttc_head import LatentTTCHead

OmegaConf.register_new_resolver("eval", eval)

@torch.no_grad()
def evaluate_diamond_ttc(
    checkpoint_path: str,
    ttc_model_path: str,
    dataset_path: str,
    mode: str = "latent",
    num_episodes: int = 5,
    rollout_steps: int = 30,
    dt: float = 0.1,
    max_ttc: float = 5.0,
    output_dir: str = "visualizations"
):
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
                if "max_ttc" in ckpt:
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
        except ImportError:
            print("VideoTTCPredictor not found, defaulting to latent mode.")
        except (ImportError, ModuleNotFoundError):
            print("Legacy VideoTTCPredictor not found, defaulting to latent mode.")

    # 3. Setup Dataset
    test_dataset_path = Path(dataset_path) / "test"
    if not test_dataset_path.exists():
        test_dataset_path = Path(dataset_path)
    print(f"Loading dataset from {test_dataset_path}...")
    test_dataset = Dataset(test_dataset_path, "test_dataset")
    test_dataset.load_from_default_path()
    
    seq_len = num_cond + rollout_steps
    batch_sampler = BatchSampler(test_dataset, rank=0, world_size=1, batch_size=1, seq_length=seq_len, sample_weights=None)
    data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_segments_to_batch)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Rollout & Predict
    print(f"\nStarting DIAMOND Rollout & TTC Evaluation over {num_episodes} episodes...")
    data_iterator = iter(data_loader)
    
    step_maes = [[] for _ in range(rollout_steps)]
    step_mses = [[] for _ in range(rollout_steps)]
    all_records = []
    all_gt = []
    all_pred = []
    
    for ep_idx in range(num_episodes):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        obs = batch.obs.to(device) # (1, seq_len, 3, H, W) range [-1, 1]
        act = batch.act.to(device) # (1, seq_len)
        end = batch.end.to(device) # (1, seq_len)
        
        # Calculate ground truth TTC for each future step
        crash_indices = torch.where(end[0] == 1)[0]
        has_crash = len(crash_indices) > 0
        crash_idx = crash_indices[0].item() if has_crash else -1
        
        history_obs = obs[:, :num_cond].clone()
        history_act = act[:, :num_cond].clone()
        
        generated_obs_list = [history_obs[:, i] for i in range(num_cond)]
        predicted_ttc_list = []
        gt_ttc_list = []
        hx_cx = None
        
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

        print(f"\nEpisode {ep_idx + 1}/{num_episodes}: Evaluating {rollout_steps}-step horizon...")
        for step in range(rollout_steps):
            curr_idx = num_cond + step - 1
            curr_act = act[:, curr_idx : curr_idx + 1]
            
            # Ground truth TTC
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
            gt_ttc_list.append(true_ttc)
            
            # TTC Prediction from Latent or Generated Frames
            if mode == "latent":
                z = agent.denoiser.extract_latent(history_obs, history_act)
                pred_ttc, hx_cx = ttc_predictor(z.unsqueeze(1), hx_cx)
                pred_val = pred_ttc.squeeze().item()
            else:
                # Legacy pixel rollout mode
                pred_val = 0.0
                
            predicted_ttc_list.append(pred_val)
            
            error_mae = abs(pred_val - true_ttc)
            error_mse = (pred_val - true_ttc) ** 2
            step_maes[step].append(error_mae)
            step_mses[step].append(error_mse)
            all_records.append({
                "episode": ep_idx + 1,
                "lookahead_step": step + 1,
                "ground_truth_ttc": round(true_ttc, 4),
                "predicted_ttc": round(pred_val, 4),
                "abs_error": round(error_mae, 4),
                "squared_error": round(error_mse, 4),
            })
            
            # Rollout step in world model
            next_obs, _ = sampler.sample(history_obs, history_act)
            generated_obs_list.append(next_obs)
            
            history_obs = torch.cat([history_obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
            history_act = torch.cat([history_act[:, 1:], curr_act], dim=1)
            
        all_gt.append(gt_ttc_list)
        all_pred.append(predicted_ttc_list)

        # Plot episode TTC trajectory
        plt.figure(figsize=(9, 4))
        plt.plot(range(1, rollout_steps + 1), gt_ttc_list, 'g--', label="Ground Truth TTC", linewidth=2)
        plt.plot(range(1, rollout_steps + 1), predicted_ttc_list, 'b-', label=f"Predicted TTC ({mode.capitalize()})", linewidth=2)
        plt.title(f"Episode {ep_idx + 1}: TTC Prediction Trajectory")
        plt.xlabel("Lookahead Step")
        plt.ylabel("TTC (seconds)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        ep_plot_path = os.path.join(output_dir, f"episode_{ep_idx + 1}_ttc_trajectory.png")
        plt.savefig(ep_plot_path)
        plt.close()
        print(f"Saved episode trajectory plot to {ep_plot_path}")

    # 1. Export Raw Step-by-Step Predictions CSV
    csv_path = os.path.join(output_dir, "eval_ttc_predictions.csv")
    with open(csv_path, mode="w", newline="") as f:
        fieldnames = ["episode", "lookahead_step", "ground_truth_ttc", "predicted_ttc", "abs_error", "squared_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved step-by-step predictions CSV to {csv_path}")

    # 2. Export Lookahead Horizon Metrics CSV
    horizon_csv_path = os.path.join(output_dir, "eval_horizon_metrics.csv")
    with open(horizon_csv_path, mode="w", newline="") as f:
        fieldnames = ["lookahead_step", "mean_mae", "mean_rmse", "std_mae"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(rollout_steps):
            errs = step_maes[step]
            sq_errs = step_mses[step]
            m_mae = float(np.mean(errs)) if len(errs) > 0 else 0.0
            m_rmse = float(np.sqrt(np.mean(sq_errs))) if len(sq_errs) > 0 else 0.0
            s_mae = float(np.std(errs)) if len(errs) > 0 else 0.0
            writer.writerow({
                "lookahead_step": step + 1,
                "mean_mae": round(m_mae, 4),
                "mean_rmse": round(m_rmse, 4),
                "std_mae": round(s_mae, 4)
            })
    print(f"Saved lookahead horizon metrics CSV to {horizon_csv_path}")

    # 3. Export Compressed NumPy NPZ Dataset
    npz_path = os.path.join(output_dir, "eval_data.npz")
    np.savez_compressed(
        npz_path,
        ground_truth=np.array(all_gt),
        predictions=np.array(all_pred),
        step_maes=np.array(step_maes),
        step_mses=np.array(step_mses)
    )
    print(f"Saved compressed NumPy arrays to {npz_path}")

    # 4. Compute Overall Summary Statistics and Save JSON
    if len(all_records) > 0:
        overall_mae = float(np.mean([r["abs_error"] for r in all_records]))
        overall_rmse = float(np.sqrt(np.mean([r["squared_error"] for r in all_records])))
        gt_flat = np.array([r["ground_truth_ttc"] for r in all_records])
        pred_flat = np.array([r["predicted_ttc"] for r in all_records])
        corr = float(np.corrcoef(gt_flat, pred_flat)[0, 1]) if len(gt_flat) > 1 and np.std(gt_flat) > 0 and np.std(pred_flat) > 0 else 0.0

        summary = {
            "overall_mae_seconds": round(overall_mae, 4),
            "overall_rmse_seconds": round(overall_rmse, 4),
            "pearson_correlation_r": round(corr, 4),
            "num_episodes": len(all_gt),
            "rollout_steps": rollout_steps,
            "mode": mode
        }
        json_path = os.path.join(output_dir, "eval_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved evaluation summary JSON to {json_path}")
        print(f"\n=================== EVALUATION SUMMARY ===================")
        print(f"Overall MAE:           {overall_mae:.4f}s")
        print(f"Overall RMSE:          {overall_rmse:.4f}s")
        print(f"Pearson Correlation R: {corr:.4f}")
        print(f"Total Evaluated Points:{len(all_records)}")
        print(f"==========================================================\n")

    # Plot Horizon Error Curve
    if any(len(errs) > 0 for errs in step_maes):
        avg_maes = [np.mean(errs) if len(errs) > 0 else 0.0 for errs in step_maes]
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, rollout_steps + 1), avg_maes, marker='o', linewidth=2, color='crimson')
        plt.title(f"Horizon of Predictability: TTC Error vs Lookahead Step ({mode.upper()} Mode)")
        plt.xlabel("Lookahead Step (+N steps into future)")
        plt.ylabel("TTC MAE (Seconds)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        horizon_plot_path = os.path.join(output_dir, "latent_ttc_horizon_error.png")
        plt.savefig(horizon_plot_path)
        plt.close()
        print(f"Saved Horizon Error Plot to {horizon_plot_path}")

    print("\nDIAMOND Latent TTC Evaluation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DIAMOND World Model Latent TTC Predictor")
    parser.add_argument("--checkpoint", type=str, default="diamond_highway_mcts.pt", help="Path to DIAMOND checkpoint")
    parser.add_argument("--ttc-model", type=str, default="checkpoints/best_latent_ttc.pt", help="Path to trained Latent TTC weights")
    parser.add_argument("--dataset_path", type=str, default="dataset_mcts", help="Path to dataset directory")
    parser.add_argument("--mode", type=str, default="latent", choices=["latent", "pixel"], help="Evaluation mode (latent or pixel)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--rollout_steps", type=int, default=30, help="Number of future steps to rollout using DIAMOND")
    parser.add_argument("--dt", type=float, default=0.1, help="Time delta per step in seconds")
    parser.add_argument("--max_ttc", type=float, default=None, help="Fallback max TTC in seconds (default: None for uncapped)")
    parser.add_argument("--output_dir", type=str, default="visualizations/latent_ttc", help="Directory to save output plots")
    
    args = parser.parse_args()
    evaluate_diamond_ttc(
        checkpoint_path=args.checkpoint,
        ttc_model_path=args.ttc_model,
        dataset_path=args.dataset_path,
        mode=args.mode,
        num_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        dt=args.dt,
        max_ttc=args.max_ttc,
        output_dir=args.output_dir
    )
