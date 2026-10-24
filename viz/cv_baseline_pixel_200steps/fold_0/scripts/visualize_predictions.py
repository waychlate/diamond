import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

import sys
sys.path.append('src')

from agent import Agent
from data import Dataset, BatchSampler, collate_segments_to_batch
from models.diffusion import DiffusionSampler

OmegaConf.register_new_resolver("eval", eval)

@torch.no_grad()
def generate_visual_comparisons(cfg: DictConfig, checkpoint_path: str, dataset_path: str, output_dir: Path, num_samples: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running visualization on {device}")

    # 1. Load Agent/Model
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    
    print(f"Loading checkpoint from {checkpoint_path}")
    agent.load(checkpoint_path)

    # 2. Setup Dataset (Test Set)
    test_dataset_path = Path(dataset_path) / "test"
    test_dataset = Dataset(test_dataset_path, "test_dataset")
    test_dataset.load_from_default_path()
    
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    seq_len = num_cond + 1
    
    batch_sampler = BatchSampler(test_dataset, rank=0, world_size=1, batch_size=1, seq_length=seq_len, sample_weights=None)
    data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_segments_to_batch)
    
    # 3. Setup Sampler
    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_iterator = iter(data_loader)
    count = 0
    
    print(f"Generating {num_samples} comparisons...")
    while count < num_samples:
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        obs = batch.obs.to(device)
        act = batch.act.to(device)
        
        history_obs = obs[:, :num_cond] 
        history_act = act[:, :num_cond]
        ground_truth_next_obs = obs[:, -1]
        
        predicted_next_obs, _ = sampler.sample(history_obs, history_act)
        
        # Denormalize [-1, 1] to [0, 255]
        gt_img = ((ground_truth_next_obs[0].clamp(-1, 1) + 1) / 2 * 255).byte().cpu().permute(1, 2, 0).numpy()
        pred_img = ((predicted_next_obs[0].clamp(-1, 1) + 1) / 2 * 255).byte().cpu().permute(1, 2, 0).numpy()
        
        # PIL Images
        gt_pil = Image.fromarray(gt_img)
        pred_pil = Image.fromarray(pred_img)
        
        w, h = gt_pil.size
        # Stack them horizontally with a spacer
        spacing = 10
        label_margin = 12
        label_height = 15
        
        # Total size
        total_width = w + spacing + w
        total_height = h + label_margin + label_height
        combined = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))
        draw = ImageDraw.Draw(combined)
        
        # Paste images at the top (y=0)
        combined.paste(gt_pil, (0, 0))
        combined.paste(pred_pil, (w + spacing, 0))
        
        # Draw labels below the images
        draw.text((5, h + label_margin), "GROUND TRUTH", fill=(255, 255, 255))
        draw.text((w + spacing + 5, h + label_margin), "DIAMOND PREDICTION", fill=(255, 255, 255))
        
        out_path = output_dir / f"comparison_{count + 1}.png"
        combined.save(out_path)
        print(f"Saved: {out_path}")
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to state.pt")
    parser.add_argument("--dataset_path", type=str, default="dataset_mcts", help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save comparison images")
    parser.add_argument("--samples", type=int, default=5, help="Number of comparison samples to generate")
    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")
        
    generate_visual_comparisons(cfg, args.checkpoint, args.dataset_path, Path(args.output_dir), args.samples)
