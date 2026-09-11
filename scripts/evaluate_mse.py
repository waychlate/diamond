import time
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
def evaluate_pure_mse(
    cfg: DictConfig,
    checkpoint_path: str,
    dataset_path: str,
    num_samples: int = 500,
    batch_size: int = 32,
    num_workers: int = 4,
    steps: int = None,
    use_fp16: bool = True,
    compile_model: bool = False,
    trt_engine: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print(" DIAMOND Edge Evaluation Benchmark (MSE & Latency)")
    print("=" * 60)
    print(f" Device:          {device}")
    print(f" Batch Size:      {batch_size}")
    print(f" FP16 Mixed Prec: {use_fp16 and device.type == 'cuda'}")
    print(f" TRT Engine:      {trt_engine if trt_engine else 'None (PyTorch Runtime)'}")

    # 1. Load Agent/Model
    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    agent.load(checkpoint_path)

    # Optional: Swap UNet inner model with TensorRT Engine
    if trt_engine is not None:
        try:
            from models.diffusion.trt_inner_model import TRTInnerModel
            print(f"Loading TensorRT Engine from {trt_engine}...")
            agent.denoiser.inner_model = TRTInnerModel(trt_engine, device=device)
            print("TensorRT UNet Engine successfully attached to DIAMOND Denoiser!")
        except Exception as e:
            print(f"Failed to load TensorRT Engine: {e}")
            print("Falling back to PyTorch UNet.")

    elif compile_model and hasattr(torch, "compile"):
        print("Compiling denoiser inner model with torch.compile...")
        agent.denoiser.inner_model = torch.compile(agent.denoiser.inner_model)

    # 2. Setup Dataset (Test Set)
    test_dataset_path = Path(dataset_path) / "test"
    test_dataset = Dataset(test_dataset_path, "test_dataset")
    test_dataset.load_from_default_path()
    
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning if hasattr(cfg.agent.denoiser.inner_model, "num_steps_conditioning") else 4
    seq_len = num_cond + 1
    
    batch_sampler = BatchSampler(
        test_dataset,
        rank=0,
        world_size=1,
        batch_size=batch_size,
        seq_length=seq_len,
        sample_weights=None,
    )
    data_loader = DataLoader(
        test_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_segments_to_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # 3. Setup Sampler
    sampler_cfg = cfg.world_model_env.diffusion_sampler
    if steps is not None and steps > 0:
        sampler_cfg.num_steps_denoising = steps
        print(f" Diffusion Steps: {steps} (Overridden via CLI)")
    else:
        print(f" Diffusion Steps: {sampler_cfg.num_steps_denoising} (order={sampler_cfg.order})")

    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    total_squared_error = 0.0
    total_evaluated_samples = 0
    total_inference_time = 0.0
    
    print(f"\nStarting evaluation over {num_samples} samples...")
    pbar = tqdm(total=num_samples, desc="Evaluating")
    
    data_iterator = iter(data_loader)
    amp_enabled = use_fp16 and (device.type == "cuda") and (trt_engine is None)
    
    # Warmup GPU
    if device.type == "cuda":
        dummy_obs = torch.randn(1, num_cond, 3, 48, 320, device=device)
        dummy_act = torch.zeros(1, num_cond, device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            sampler.sample(dummy_obs, dummy_act)
        torch.cuda.synchronize()

    while total_evaluated_samples < num_samples:
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        current_b = batch.obs.shape[0]
        if total_evaluated_samples + current_b > num_samples:
            slice_len = num_samples - total_evaluated_samples
            obs = batch.obs[:slice_len].to(device, non_blocking=True)
            act = batch.act[:slice_len].to(device, non_blocking=True)
            current_b = slice_len
        else:
            obs = batch.obs.to(device, non_blocking=True)
            act = batch.act.to(device, non_blocking=True)
        
        history_obs = obs[:, :num_cond] 
        history_act = act[:, :num_cond]
        ground_truth_next_obs = obs[:, -1]  # (B, 3, H, W)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            predicted_next_obs, _ = sampler.sample(history_obs, history_act)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_time = t1 - t0
        total_inference_time += batch_time

        batch_mse_per_sample = torch.nn.functional.mse_loss(
            predicted_next_obs, ground_truth_next_obs, reduction="none"
        ).flatten(start_dim=1).mean(dim=1)
        
        total_squared_error += batch_mse_per_sample.sum().item()
        total_evaluated_samples += current_b
        pbar.update(current_b)
        
    pbar.close()
    
    if total_evaluated_samples > 0:
        final_mse = total_squared_error / total_evaluated_samples
        avg_latency_ms = (total_inference_time / total_evaluated_samples) * 1000.0
        fps = total_evaluated_samples / total_inference_time

        print("\n" + "=" * 60)
        print(" RESULTS SUMMARY")
        print("=" * 60)
        print(f" Evaluated Samples:      {total_evaluated_samples}")
        print(f" Final Pure MSE Score:   {final_mse:.6f}")
        print(f" Average Latency / Frame: {avg_latency_ms:.2f} ms")
        print(f" Throughput:             {fps:.2f} frames/sec (FPS)")
        print("=" * 60)
        return final_mse
    else:
        print("\nNo samples evaluated.")
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient MSE & Latency Evaluation for DIAMOND on Jetson / GPU")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--dataset_path", type=str, default="dataset_mcts", help="Path to dataset directory")
    parser.add_argument("--samples", type=int, default=500, help="Number of test samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation (default: 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--steps", type=int, default=None, help="Override denoising diffusion steps (e.g. 1, 2, 3)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 mixed precision")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--trt_engine", type=str, default=None, help="Path to compiled TensorRT .engine file")
    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")
        
    evaluate_pure_mse(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        use_fp16=not args.no_fp16,
        compile_model=args.compile,
        trt_engine=args.trt_engine,
    )
