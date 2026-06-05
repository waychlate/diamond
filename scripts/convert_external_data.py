import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path to import DIAMOND modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data.episode import Episode
from data.dataset import Dataset

def convert_data(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        print(f"\nConverting {split} split...")
        src_split_dir = src_dir / f"processed_{split}"
        if not src_split_dir.exists():
            print(f"Warning: {src_split_dir} does not exist, skipping.")
            continue
            
        dst_split_dir = dst_dir / split
        
        # Initialize Dataset (this will manage the filesystem hierarchy)
        dataset = Dataset(dst_split_dir, name=f"{split}_dataset")
        dataset.clear()

        # Find all episodes
        csv_files = sorted(list(src_split_dir.glob("episode_*_data.csv")))
        
        for csv_file in tqdm(csv_files, desc=f"Processing {split} episodes"):
            episode_id_str = csv_file.stem.split("_")[1]
            npz_file = src_split_dir / f"episode_{episode_id_str}_visuals.npz"
            
            if not npz_file.exists():
                continue
                
            # Load CSV and NPZ
            try:
                df = pd.read_csv(csv_file)
                visuals = np.load(npz_file)["visuals"] # (T, H, W, C)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue
            
            T = visuals.shape[0]
            if len(df) != T:
                T = min(T, len(df))
                visuals = visuals[:T]
                df = df.iloc[:T]

            # Safety check: need at least 2 frames for 1 transition
            if T <= 1:
                continue

            # 1. Process Observations (normalize to [-1, 1], permute to BCHW)
            # Note: Dataset.add_episode calls Episode.save which handles the uint8 conversion for disk.
            obs = torch.from_numpy(visuals[:T-1]).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
            
            # 2. Process Actions and Rewards
            act = torch.from_numpy(df.action.values[:T-1].astype(np.int64))
            rew = torch.from_numpy(df.reward.values[:T-1].astype(np.float32))
            
            # 3. Process Termination Flags
            # Map 'done' column to end tensor
            done_values = df.done.values[:T-1]
            if done_values.dtype == bool:
                end = torch.from_numpy(done_values.astype(np.uint8))
            else:
                end = torch.from_numpy((done_values == True).astype(np.uint8))
            
            # Truncation: In these logs, if it didn't 'done', it likely timed out at step 100
            trunc = torch.zeros_like(end)
            if not end[-1]:
                trunc[-1] = 1
                
            # 4. Info block with final_observation (crucial for World Model next-state prediction)
            info = {
                "final_observation": torch.from_numpy(visuals[T-1]).permute(2, 0, 1).float().div(255).mul(2).sub(1)
            }
            
            # 5. Create and Add Episode
            episode = Episode(obs, act, rew, end, trunc, info)
            dataset.add_episode(episode)
            
        # Finalize dataset metadata
        dataset.save_to_default_path()
        print(f"Finished {split} split. {dataset}")

if __name__ == "__main__":
    SRC = "/blue/iruchkin/khek.do/final_output"
    DST = "/blue/iruchkin/khek.do/dataset_mcts"
    convert_data(SRC, DST)
