import os
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    TRT_AVAILABLE = False


class TRTInnerModel(nn.Module):
    """
    Drop-in replacement for InnerModel that executes a compiled TensorRT engine
    with zero-copy PyTorch CUDA tensor bindings.
    """
    def __init__(self, engine_path: str, device: torch.device = None) -> None:
        super().__init__()
        if not TRT_AVAILABLE:
            raise ImportError(
                "TensorRT is not installed. Please install TensorRT or run on an NVIDIA Jetson / GPU with TensorRT."
            )
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        self.engine_path = engine_path
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize TensorRT Logger and Runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self._inspect_engine_bindings()

    def _inspect_engine_bindings(self):
        self.is_v3 = hasattr(self.engine, "num_io_tensors")
        self.input_names = ["noisy_next_obs", "c_noise", "obs", "act"]
        self.output_names = ["denoised_output"]

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        b = noisy_next_obs.shape[0]
        h, w = noisy_next_obs.shape[2], noisy_next_obs.shape[3]
        out_channels = 3  # RGB image channels

        noisy_next_obs = noisy_next_obs.contiguous().to(self.device)
        c_noise = c_noise.contiguous().to(self.device)
        obs = obs.contiguous().to(self.device)
        act = act.contiguous().to(self.device, dtype=torch.long)

        output = torch.empty((b, out_channels, h, w), device=self.device, dtype=noisy_next_obs.dtype)

        if self.is_v3:
            # Modern TensorRT 8.6+ / 10+ API
            self.context.set_input_shape("noisy_next_obs", noisy_next_obs.shape)
            self.context.set_input_shape("c_noise", c_noise.shape)
            self.context.set_input_shape("obs", obs.shape)
            self.context.set_input_shape("act", act.shape)

            self.context.set_tensor_address("noisy_next_obs", noisy_next_obs.data_ptr())
            self.context.set_tensor_address("c_noise", c_noise.data_ptr())
            self.context.set_tensor_address("obs", obs.data_ptr())
            self.context.set_tensor_address("act", act.data_ptr())
            self.context.set_tensor_address("denoised_output", output.data_ptr())

            stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
            self.context.execute_async_v3(stream_handle=stream_ptr)
        else:
            # Classic TensorRT v2 API
            bindings = [
                noisy_next_obs.data_ptr(),
                c_noise.data_ptr(),
                obs.data_ptr(),
                act.data_ptr(),
                output.data_ptr(),
            ]
            self.context.execute_v2(bindings=bindings)

        return output

