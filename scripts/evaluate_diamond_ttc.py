import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

# Add src to sys.path for DIAMOND imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add TTC_Prediction to sys.path for TTC model imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "TTC_Prediction"))

from agent import Agent
from data import Dataset, BatchSampler, collate_segments_to_batch
from models.diffusion import DiffusionSampler
from scripts.model import VideoTTCPredictor

OmegaConf.register_new_resolver("eval", eval)

@torch.no_grad()
def evaluate_diamond_ttc(
    checkpoint_path: str,
    ttc_model_path: str,
    dataset_path: str,
    num_episodes: int = 5,
    rollout_steps: int = 30,
    output_dir: str = "visualizations"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Hydra Config & DIAMOND Agent
    config_dir = "../config"
    with initialize(version_base="1.3", config_path=config_dir):
        cfg = compose(config_name="trainer", overrides=["env=highway"])
        
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    print("Instantiating DIAMOND Agent...")
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    
    print(f"Loading DIAMOND checkpoint from {checkpoint_path}...")
    agent.load(checkpoint_path)
    
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    print(f"DIAMOND conditioning steps: {num_cond}")
    
    # 2. Load TTC Predictor Model
    print(f"Loading TTC Predictor from {ttc_model_path}...")
    ttc_predictor = VideoTTCPredictor(
        hidden_dim=128,
        action_dim=16,
        use_actions=True,
        num_layers=1,
        backbone_type="custom",
        in_channels=9 # 3-frame stacked (9 channels)
    ).to(device).eval()
    
    if os.path.exists(ttc_model_path):
        checkpoint = torch.load(ttc_model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            ttc_predictor.load_state_dict(checkpoint["model_state_dict"])
        else:
            ttc_predictor.load_state_dict(checkpoint)
        print("TTC Predictor weights loaded successfully.")
    else:
        print(f"Warning: {ttc_model_path} not found. Proceeding with initialized weights for structure verification.")

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
    print(f"\nStarting DIAMOND Rollout & TTC Prediction over {num_episodes} episodes...")
    data_iterator = iter(data_loader)
    
    all_step_errors = [[] for _ in range(rollout_steps)]
    
    for ep_idx in range(num_episodes):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        obs = batch.obs.to(device) # (1, seq_len, 3, H, W) range [-1, 1]
        act = batch.act.to(device) # (1, seq_len)
        
        # Autoregressive Rollout using DIAMOND
        history_obs = obs[:, :num_cond].clone()
        history_act = act[:, :num_cond].clone()
        
        generated_obs_list = [history_obs[:, i] for i in range(num_cond)]
        
        print(f"\nEpisode {ep_idx + 1}/{num_episodes}: Generating {rollout_steps}-step rollout...")
        for step in range(rollout_steps):
            curr_act = act[:, num_cond + step - 1 : num_cond + step]
            # Input to sampler expects history_obs, history_act
            next_obs, _ = sampler.sample(history_obs, history_act)
            generated_obs_list.append(next_obs)
            
            # Slide window for next step
            history_obs = torch.cat([history_obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
            history_act = torch.cat([history_act[:, 1:], curr_act], dim=1)
            
        generated_obs_seq = torch.stack(generated_obs_list, dim=1) # (1, num_cond + rollout_steps, 3, H, W)
        
        # Plot generated vs ground truth frame sample
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            idx = num_cond + i * (rollout_steps // 5)
            # Ground truth
            gt_img = obs[0, idx].permute(1, 2, 0).cpu().numpy()
            gt_img = np.clip((gt_img + 1.0) / 2.0, 0.0, 1.0)
            axes[0, i].imshow(gt_img)
            axes[0, i].set_title(f"GT Step {idx}")
            axes[0, i].axis("off")
            
            # DIAMOND Rollout
            gen_img = generated_obs_seq[0, idx].permute(1, 2, 0).cpu().numpy()
            gen_img = np.clip((gen_img + 1.0) / 2.0, 0.0, 1.0)
            axes[1, i].imshow(gen_img)
            axes[1, i].set_title(f"DIAMOND Step {idx}")
            axes[1, i].axis("off")
            
        plt.suptitle(f"Episode {ep_idx + 1}: Real (Top) vs DIAMOND Rollout (Bottom)")
        plt.tight_layout()
        save_plot_path = os.path.join(output_dir, f"diamond_rollout_ep{ep_idx + 1}.png")
        plt.savefig(save_plot_path)
        plt.close()
        print(f"Saved rollout visualization to {save_plot_path}")

    print("\nDIAMOND + TTC Evaluation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DIAMOND World Model Rollouts with TTC Predictor")
    parser.add_argument("--checkpoint", type=str, default="diamond_highway_mcts.pt", help="Path to DIAMOND checkpoint")
    parser.add_argument("--ttc-model", type=str, default="../TTC_Prediction/best_model.pth", help="Path to trained TTC model weights")
    parser.add_argument("--dataset_path", type=str, default="dataset_mcts", help="Path to dataset directory")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--rollout_steps", type=int, default=30, help="Number of future steps to rollout using DIAMOND")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save output plots")
    
    args = parser.parse_args()
    evaluate_diamond_ttc(
        checkpoint_path=args.checkpoint,
        ttc_model_path=args.ttc_model,
        dataset_path=args.dataset_path,
        num_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        output_dir=args.output_dir
    )
