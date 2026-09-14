import os
import sys
import csv
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

# Add src to sys.path for DIAMOND modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add TTC_Prediction to sys.path for TTC model imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "TTC_Prediction"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "TTC_Prediction"))

from agent import Agent
from models.diffusion import DiffusionSampler
try:
    from scripts.model import VideoTTCPredictor
except ImportError:
    from model import VideoTTCPredictor

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_episode_raw_data(data_dir: str, ep_idx: int):
    """
    Loads raw episode CSV and visuals NPZ if available.
    Returns: (raw_frames, ttc_targets, actions)
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_data.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*_visuals.npz")))
    if not npz_files:
        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
    if ep_idx >= len(csv_files) or ep_idx >= len(npz_files):
        raise IndexError(f"Episode index {ep_idx} out of range (found {len(csv_files)} CSVs, {len(npz_files)} NPZs)")
        
    csv_path = csv_files[ep_idx]
    npz_path = npz_files[ep_idx]
    
    # Read CSV
    ttc_list = []
    act_list = []
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttc_val = float(row.get('obs_ttc', row.get('ttc', 0.0)))
            act_val = int(row.get('action', row.get('actions', 0)))
            ttc_list.append(ttc_val)
            act_list.append(act_val)
            
    with np.load(npz_path) as npz:
        raw_visuals = npz['visuals'] # (T, H, W, C)
        
    T = min(len(ttc_list), raw_visuals.shape[0])
    return raw_visuals[:T], np.array(ttc_list[:T], dtype=np.float32), np.array(act_list[:T], dtype=np.int64)

def preprocess_for_diamond(raw_visuals_np: np.ndarray) -> torch.Tensor:
    """
    Converts raw visuals (T, 150, 600, 3) to DIAMOND format (T, 3, 48, 320) in [-1, 1].
    """
    T = raw_visuals_np.shape[0]
    frames = torch.from_numpy(raw_visuals_np).float() / 255.0 # (T, H, W, C)
    frames = frames.permute(0, 3, 1, 2) # (T, C, H, W)
    
    processed = []
    for t in range(T):
        cropped = TF.crop(frames[t], top=60, left=0, height=90, width=600)
        resized = TF.resize(cropped, size=[48, 320], antialias=True)
        processed.append(resized)
        
    # Shape: (T, 3, 48, 320) in [-1, 1]
    return torch.stack(processed).mul(2.0).sub(1.0)

def preprocess_for_ttc(frame_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Converts frame tensor (from DIAMOND format in [-1, 1], shape (B, C, 48, 320) or (C, 48, 320))
    to TTC model input normalized (B, C, 64, 256).
    """
    if frame_tensor.ndim == 3:
        frame_tensor = frame_tensor.unsqueeze(0)
        
    # Scale [-1, 1] -> [0, 1]
    img_01 = torch.clamp((frame_tensor + 1.0) / 2.0, 0.0, 1.0)
    # Resize to (64, 256)
    img_resized = F.interpolate(img_01, size=(64, 256), mode='bilinear', align_corners=False)
    # Normalize with ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_resized - mean) / std

@torch.no_grad()
def evaluate_diamond_ttc_lookahead(
    checkpoint_path: str,
    ttc_model_path: str,
    dataset_path: str,
    num_episodes: int = 10,
    context_frames: int = 20,
    lookahead_steps: int = 30,
    output_dir: str = "visualizations/ttc_lookahead_eval"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running TTC Evaluation on {device} ===")
    print(f"Initial Context: {context_frames} frames")
    print(f"Lookahead Horizon: +1 to +{lookahead_steps} steps")
    print(f"Number of Episodes: {num_episodes}")
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load DIAMOND Model
    print(f"\nLoading DIAMOND World Model from: {checkpoint_path}")
    config_dir = "../config"
    with initialize(version_base="1.3", config_path=config_dir):
        cfg = compose(config_name="trainer", overrides=["env=highway"])
        
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(checkpoint_path)
    
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    print(f"DIAMOND conditioning history frames: {num_cond}")

    # 2. Load TTC Predictor Model
    print(f"\nLoading TTC Predictor from: {ttc_model_path}")
    ttc_predictor = VideoTTCPredictor(
        hidden_dim=256,
        action_dim=16,
        use_actions=False,
        num_layers=2,
        backbone_type="resnet18",
        in_channels=3
    ).to(device).eval()
    
    checkpoint = torch.load(ttc_model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        ttc_predictor.load_state_dict(checkpoint["model_state_dict"])
    else:
        ttc_predictor.load_state_dict(checkpoint)
    print("TTC Predictor weights loaded successfully.")

    # 3. Locate Dataset
    test_dir = Path(dataset_path)
    if (test_dir / "test").exists():
        test_dir = test_dir / "test"
    print(f"\nEvaluating dataset from: {test_dir}")

    # Storage for per-episode per-step errors
    # shape: (num_episodes, lookahead_steps)
    episode_squared_errors = []
    episode_abs_errors = []
    detailed_records = []

    total_eval_episodes = 0

    for ep_idx in range(num_episodes):
        try:
            raw_visuals, ttc_targets, actions = load_episode_raw_data(str(test_dir), ep_idx)
        except Exception as e:
            print(f"Could not load episode {ep_idx}: {e}")
            break

        total_steps_needed = context_frames + lookahead_steps
        if len(ttc_targets) < total_steps_needed:
            print(f"Skipping episode {ep_idx + 1}: length {len(ttc_targets)} < required {total_steps_needed}")
            continue

        print(f"\n[Episode {ep_idx + 1}/{num_episodes}] Processing ({context_frames} context + {lookahead_steps} lookahead steps)...")
        
        # Preprocess episode frames to DIAMOND tensor format: (T, 3, 48, 320)
        diamond_obs = preprocess_for_diamond(raw_visuals[:total_steps_needed]).to(device)
        diamond_act = torch.from_numpy(actions[:total_steps_needed]).to(device).unsqueeze(0) # (1, T)

        # Autoregressive Rollout using DIAMOND
        # Context frames: 0 .. context_frames-1
        # DIAMOND conditioning: last `num_cond` frames from the context
        history_obs = diamond_obs[context_frames - num_cond : context_frames].unsqueeze(0).clone() # (1, num_cond, 3, 48, 320)
        history_act = diamond_act[:, context_frames - num_cond : context_frames].clone() # (1, num_cond)
        
        # Store full frame sequence: first context_frames real, then lookahead_steps generated
        generated_frames = [diamond_obs[i] for i in range(context_frames)]

        for step in range(lookahead_steps):
            curr_step_idx = context_frames + step
            # Action taken leading to next state
            curr_act = diamond_act[:, curr_step_idx - 1 : curr_step_idx]
            
            # Predict next frame using DIAMOND diffusion sampler
            next_obs, _ = sampler.sample(history_obs, history_act) # (1, 3, 48, 320)
            generated_frames.append(next_obs.squeeze(0))
            
            # Update rolling conditioning window for DIAMOND
            history_obs = torch.cat([history_obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
            history_act = torch.cat([history_act[:, 1:], curr_act], dim=1)

        # Now evaluate TTC at each lookahead step 1 .. lookahead_steps
        ep_sq_errs = []
        ep_abs_errs = []

        for step in range(1, lookahead_steps + 1):
            target_step_idx = context_frames + step - 1
            gt_ttc = float(ttc_targets[target_step_idx])

            # Take the sliding 20-frame context window ending at target_step_idx
            # Window indices: [target_step_idx - context_frames + 1 : target_step_idx + 1]
            window_frames = generated_frames[target_step_idx - context_frames + 1 : target_step_idx + 1]
            window_tensor = torch.stack(window_frames, dim=0) # (20, 3, 48, 320)

            # Preprocess window for TTC model: (1, 20, 3, 64, 256)
            ttc_input = preprocess_for_ttc(window_tensor, device).unsqueeze(0)

            # Predict TTC
            pred_ttc_tensor = ttc_predictor(ttc_input)
            pred_ttc = float(pred_ttc_tensor.squeeze().item())

            sq_err = (pred_ttc - gt_ttc) ** 2
            abs_err = abs(pred_ttc - gt_ttc)

            ep_sq_errs.append(sq_err)
            ep_abs_errs.append(abs_err)

            detailed_records.append({
                "episode": ep_idx + 1,
                "lookahead_step": step,
                "ground_truth_ttc": round(gt_ttc, 4),
                "predicted_ttc": round(pred_ttc, 4),
                "squared_error": round(sq_err, 4),
                "abs_error": round(abs_err, 4)
            })

        episode_squared_errors.append(ep_sq_errs)
        episode_abs_errors.append(ep_abs_errs)
        total_eval_episodes += 1
        print(f"Episode {ep_idx + 1} Avg MSE over lookahead: {np.mean(ep_sq_errs):.4f} (MAE: {np.mean(ep_abs_errs):.4f}s)")

    if total_eval_episodes == 0:
        print("Error: No episodes were successfully evaluated.")
        return

    # Convert to NumPy array: shape (num_episodes, lookahead_steps)
    sq_errs_arr = np.array(episode_squared_errors)
    abs_errs_arr = np.array(episode_abs_errors)

    mean_mse_per_step = np.mean(sq_errs_arr, axis=0)
    std_mse_per_step = np.std(sq_errs_arr, axis=0)
    sem_mse_per_step = std_mse_per_step / np.sqrt(total_eval_episodes)

    mean_mae_per_step = np.mean(abs_errs_arr, axis=0)
    std_mae_per_step = np.std(abs_errs_arr, axis=0)

    # 4. Save CSV Reports
    detailed_csv_path = os.path.join(output_dir, "ttc_predictions_per_step.csv")
    with open(detailed_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "lookahead_step", "ground_truth_ttc", "predicted_ttc", "squared_error", "abs_error"])
        writer.writeheader()
        writer.writerows(detailed_records)
    print(f"\nSaved step-by-step predictions to: {detailed_csv_path}")

    metrics_csv_path = os.path.join(output_dir, "ttc_horizon_mse_metrics.csv")
    with open(metrics_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lookahead_step", "mean_mse", "std_mse", "sem_mse", "mean_mae", "std_mae"])
        for k in range(lookahead_steps):
            writer.writerow([
                k + 1,
                round(float(mean_mse_per_step[k]), 4),
                round(float(std_mse_per_step[k]), 4),
                round(float(sem_mse_per_step[k]), 4),
                round(float(mean_mae_per_step[k]), 4),
                round(float(std_mae_per_step[k]), 4)
            ])
    print(f"Saved aggregated metrics to: {metrics_csv_path}")

    # Summary JSON
    summary_path = os.path.join(output_dir, "eval_summary.json")
    summary_data = {
        "num_episodes": total_eval_episodes,
        "context_frames": context_frames,
        "lookahead_steps": lookahead_steps,
        "overall_mean_mse": round(float(np.mean(mean_mse_per_step)), 4),
        "overall_mean_rmse": round(float(np.sqrt(np.mean(mean_mse_per_step))), 4),
        "overall_mean_mae": round(float(np.mean(mean_mae_per_step)), 4),
        "step_1_mse": round(float(mean_mse_per_step[0]), 4),
        "step_10_mse": round(float(mean_mse_per_step[min(9, lookahead_steps - 1)]), 4),
        "step_final_mse": round(float(mean_mse_per_step[-1]), 4)
    }
    with open(summary_path, mode='w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved evaluation summary to: {summary_path}")

    # 5. Generate Professional MSE Graph
    steps_x = np.arange(1, lookahead_steps + 1)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(steps_x, mean_mse_per_step, color='#1f77b4', linewidth=2.5, marker='o', markersize=5, label=f'Avg MSE ({total_eval_episodes} episodes)')
    plt.fill_between(steps_x, np.maximum(0, mean_mse_per_step - sem_mse_per_step), mean_mse_per_step + sem_mse_per_step, color='#1f77b4', alpha=0.2, label='±1 SEM (Standard Error)')
    
    # Also plot individual episode traces softly in the background
    for ep_i in range(total_eval_episodes):
        plt.plot(steps_x, sq_errs_arr[ep_i], color='gray', alpha=0.18, linewidth=1.0)
    
    plt.title(f"TTC Prediction Error vs Lookahead Horizon (Context: {context_frames} Frames)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Lookahead Step (+N steps into DIAMOND rollout)", fontsize=11, labelpad=8)
    plt.ylabel("TTC Mean Squared Error (seconds²)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "ttc_mse_lookahead_10episodes.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n=======================================================")
    print(f"Saved 10-Episode Average MSE Graph to: {plot_path}")
    print(f"Overall Average MSE across lookahead: {summary_data['overall_mean_mse']:.4f} s²")
    print(f"Overall Average MAE across lookahead: {summary_data['overall_mean_mae']:.4f} s")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TTC Prediction with 20-frame context across lookahead rollouts")
    parser.add_argument("--checkpoint", type=str, default="diamond_highway_mcts.pt", help="Path to DIAMOND checkpoint")
    parser.add_argument("--ttc-model", type=str, default="../TTC_Prediction/results/best_model.pth", help="Path to trained TTC model weights")
    parser.add_argument("--dataset_path", type=str, default="../TTC_Prediction/data/output_ttc_sorted/test", help="Path to test dataset directory")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--context_frames", type=int, default=20, help="Number of initial context frames (default: 20)")
    parser.add_argument("--lookahead_steps", type=int, default=30, help="Number of future rollout steps (default: 30)")
    parser.add_argument("--output_dir", type=str, default="visualizations/ttc_lookahead_eval", help="Directory to save output plots and reports")
    
    args = parser.parse_args()
    evaluate_diamond_ttc_lookahead(
        checkpoint_path=args.checkpoint,
        ttc_model_path=args.ttc_model,
        dataset_path=args.dataset_path,
        num_episodes=args.episodes,
        context_frames=args.context_frames,
        lookahead_steps=args.lookahead_steps,
        output_dir=args.output_dir
    )
