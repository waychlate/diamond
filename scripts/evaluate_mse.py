import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

import sys
sys.path.append('src')

from agent import Agent
from data import Dataset, BatchSampler, collate_segments_to_batch
from models.diffusion import DiffusionSampler

OmegaConf.register_new_resolver("eval", eval)

@torch.no_grad()
def evaluate_pure_mse(cfg: DictConfig, checkpoint_path: str, num_samples: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    # 1. Load Agent/Model
    # Note: num_actions needs to match the training
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    
    print(f"Loading checkpoint from {checkpoint_path}")
    agent.load(checkpoint_path)

    # 2. Setup Dataset (Test Set)
    test_dataset_path = Path(cfg.static_dataset.path) / "test"
    test_dataset = Dataset(test_dataset_path, "test_dataset")
    test_dataset.load_from_default_path()
    
    # We need sequences of length 'num_steps_conditioning + 1'
    # to have enough history to predict the next frame.
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    seq_len = num_cond + 1
    
    batch_sampler = BatchSampler(test_dataset, rank=0, world_size=1, batch_size=1, seq_length=seq_len, sample_weights=None)
    data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_segments_to_batch)
    
    # 3. Setup Sampler
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)
    
    mse_accumulator = 0.0
    count = 0
    
    print(f"Starting MSE evaluation over {num_samples} samples...")
    pbar = tqdm(total=num_samples)
    
    data_iterator = iter(data_loader)
    
    while count < num_samples:
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        # batch.obs shape: (1, num_cond + 1, 3, H, W)
        # batch.act shape: (1, num_cond + 1)
        
        obs = batch.obs.to(device)
        act = batch.act.to(device)
        
        # We use the first 'num_cond' frames as history
        # and try to predict the very last frame in the segment.
        history_obs = obs[:, :num_cond] 
        history_act = act[:, :num_cond]
        ground_truth_next_obs = obs[:, -1] # The last frame (s_{t+1})
        
        # Predict next obs
        # Note: DiffusionSampler expects (history_obs, history_act)
        # It will predict the frame corresponding to the LAST action in history_act.
        predicted_next_obs, _ = sampler.sample(history_obs, history_act)
        
        # Calculate MSE
        # Tensors are in range [-1, 1]
        mse = torch.nn.functional.mse_loss(predicted_next_obs, ground_truth_next_obs)
        
        mse_accumulator += mse.item()
        count += 1
        pbar.update(1)
        
    pbar.close()
    
    final_mse = mse_accumulator / count
    print(f"\nFinal Pure MSE Score: {final_mse:.6f}")
    return final_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to state.pt")
    parser.add_argument("--samples", type=int, default=500, help="Number of test samples to average over")
    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")
        
    evaluate_pure_mse(cfg, args.checkpoint, args.samples)
