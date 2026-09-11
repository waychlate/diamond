import os
import sys
import argparse
from pathlib import Path
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import Agent

OmegaConf.register_new_resolver("eval", eval)


def export_unet_onnx(
    checkpoint_path: str,
    output_onnx_path: str = "diamond_unet_highway.onnx",
    opset_version: int = 17,
):
    print(f"Loading DIAMOND model with Hydra config for ONNX export...")
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    num_actions = cfg.env.num_actions if "num_actions" in cfg.env else 5
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions))
    agent.eval()

    print(f"Loading weights from {checkpoint_path}...")
    agent.load(checkpoint_path)

    unet = agent.denoiser.inner_model.cpu()

    # Determine input dimensions from config
    num_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    img_channels = cfg.agent.denoiser.inner_model.img_channels
    h, w = cfg.env.train.size

    b = 1
    dummy_noisy_next_obs = torch.randn(b, img_channels, h, w, dtype=torch.float32)
    dummy_c_noise = torch.zeros(b, dtype=torch.float32)
    dummy_obs = torch.randn(b, num_cond * img_channels, h, w, dtype=torch.float32)
    dummy_act = torch.zeros(b, num_cond, dtype=torch.long)

    print(f"Exporting UNet InnerModel to {output_onnx_path} (opset {opset_version})...")
    print(f"  Inputs: noisy_next_obs: {list(dummy_noisy_next_obs.shape)}, c_noise: {list(dummy_c_noise.shape)}, obs: {list(dummy_obs.shape)}, act: {list(dummy_act.shape)}")

    export_kwargs = {
        "input_names": ["noisy_next_obs", "c_noise", "obs", "act"],
        "output_names": ["denoised_output"],
        "opset_version": opset_version,
        "do_constant_folding": True,
        "dynamic_axes": {
            "noisy_next_obs": {0: "batch_size"},
            "c_noise": {0: "batch_size"},
            "obs": {0: "batch_size"},
            "act": {0: "batch_size"},
            "denoised_output": {0: "batch_size"},
        },
    }

    try:
        torch.onnx.export(
            unet,
            (dummy_noisy_next_obs, dummy_c_noise, dummy_obs, dummy_act),
            output_onnx_path,
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            unet,
            (dummy_noisy_next_obs, dummy_c_noise, dummy_obs, dummy_act),
            output_onnx_path,
            **export_kwargs,
        )

    print(f"Successfully exported ONNX model to: {output_onnx_path}")
    print(f"File size: {os.path.getsize(output_onnx_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DIAMOND UNet InnerModel to ONNX")
    parser.add_argument("--checkpoint", type=str, default="diamond_highway_mcts.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, default="diamond_unet_highway.onnx", help="Output path for .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    export_unet_onnx(args.checkpoint, args.output, args.opset)
