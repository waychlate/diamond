import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as TF
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
        print(f"\nProcessing and converting {split} split...")
        
        # Check source directory for this split.
        # It expects raw files (episode_*_data.csv and episode_*_visuals.npz) to be in this directory.
        src_split_dir = src_dir / f"processed_{split}"
        if not src_split_dir.exists():
            # Fall back to checking sorted_output structure or similar if needed
            src_split_dir = src_dir / split
            if not src_split_dir.exists():
                print(f"Warning: neither 'processed_{split}' nor '{split}' subfolder exists in {src_dir}. Skipping split.")
                continue
            
        dst_split_dir = dst_dir / split
        
        # Initialize Dataset (this will manage the filesystem hierarchy)
        dataset = Dataset(dst_split_dir, name=f"{split}_dataset")
        dataset.clear()

        # Find all episodes by looking at CSV files
        csv_files = sorted(list(src_split_dir.glob("episode_*_data.csv")))
        
        for csv_file in tqdm(csv_files, desc=f"Processing {split} episodes"):
            episode_id_str = csv_file.name.split("_")[1]
            npz_file = src_split_dir / f"episode_{episode_id_str}_visuals.npz"
            
            if not npz_file.exists():
                continue
                
            # Load CSV and raw visuals NPZ
            try:
                df = pd.read_csv(csv_file)
                raw_data = np.load(npz_file)
                visuals = raw_data["visuals"]  # (T, H, W, C)
            except Exception as e:
                print(f"Error loading {csv_file.name} or corresponding npz: {e}")
                continue
            
            T = visuals.shape[0]
            if len(df) != T:
                T = min(T, len(df))
                visuals = visuals[:T]
                df = df.iloc[:T]

            # Safety check: need at least 2 frames for 1 transition
            if T <= 1:
                continue

            # --- Image Processing (Crop, Resize, Normalize & Permute) ---
            # 1. Convert numpy visuals to float PyTorch tensor in [0.0, 1.0] and shape (T, C, H, W)
            frames = torch.from_numpy(visuals).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)

            processed_frames = []
            for t in range(T):
                frame = frames[t]
                # Crop top=60, left=0, height=90, width=600
                cropped_frame = TF.crop(frame, top=60, left=0, height=90, width=600)
                # Resize to (48, 320)
                squeezed_frame = TF.resize(cropped_frame, size=[48, 320], antialias=True)
                processed_frames.append(squeezed_frame)

            # Stack to get (T, C, H, W) tensor and shift range from [0, 1] to [-1, 1]
            obs_all = torch.stack(processed_frames).mul(2.0).sub(1.0)

            # --- Construct DIAMOND Episode ---
            # 1. Observations: first T-1 processed frames (shape: T-1, C, H, W)
            obs = obs_all[:T-1]
            
            # 2. Actions and Rewards
            act = torch.from_numpy(df.action.values[:T-1].astype(np.int64))
            rew = torch.from_numpy(df.reward.values[:T-1].astype(np.float32))
            
            # 3. Termination Flags
            done_values = df.done.values[:T-1]
            if done_values.dtype == bool:
                end = torch.from_numpy(done_values.astype(np.uint8))
            else:
                end = torch.from_numpy((done_values == True).astype(np.uint8))
            
            # Truncation: if the episode ended without 'done' (e.g. timeout at step 100)
            trunc = torch.zeros_like(end)
            if not end[-1]:
                trunc[-1] = 1
                
            # 4. Info block with final_observation (the T-th processed frame)
            info = {
                "final_observation": obs_all[T-1]
            }
            
            # 5. Create and Add Episode (Dataset.add_episode handles compressing back to uint8 on disk)
            episode = Episode(obs, act, rew, end, trunc, info)
            dataset.add_episode(episode)
            
        # Finalize dataset metadata
        dataset.save_to_default_path()
        print(f"Finished {split} split. {dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge image processing and external data conversion for DIAMOND dataset format.")
    parser.add_argument("--src_dir", type=str, default="", help="Source directory containing episode CSV and visuals NPZ files.")
    parser.add_argument("--dst_dir", type=str, default="", help="Destination folder to write the preprocessed DIAMOND .pt files.")
    args = parser.parse_args()

    convert_data(args.src_dir, args.dst_dir)
