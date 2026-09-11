# NVIDIA Jetson Optimization & Evaluation Guide for DIAMOND

This guide explains how to optimize and benchmark the **DIAMOND** diffusion world model (`diamond_highway_mcts.pt`) on **NVIDIA Jetson** devices (Orin Nano, Orin NX, AGX Orin, Xavier NX) to achieve real-time inference and measure pure MSE and latency for model comparisons.

---

## 🛠️ Step 1: Maximize Jetson Performance Clocks

Jetson devices boot in low-power modes (15W/25W) by default. Lock the hardware clocks to their maximum frequencies:

```bash
# 1. Set power mode to MAXN (Maximum Performance)
sudo nvpmodel -m 0

# 2. Lock CPU, GPU, and memory controller (EMC) clocks to maximum
sudo jetson_clocks
```

---

## 🏎️ Step 2: Maximum Performance via TensorRT (FP16 Engine)

To achieve the lowest latency and highest throughput, compile DIAMOND's UNet into a TensorRT engine.

### 2.1 Export UNet to ONNX

Export the UNet denoiser inner model from your checkpoint with dynamic batching:

```bash
python scripts/export_onnx.py \
  --checkpoint diamond_highway_mcts.pt \
  --output diamond_unet_highway.onnx
```

### 2.2 Build the TensorRT FP16 Engine on the Jetson

Run the engine builder script on the target Jetson:

```bash
bash scripts/build_trt_engine.sh diamond_unet_highway.onnx diamond_unet_fp16.engine fp16
```

_(This compiles layer fusions—Conv + Bias + SiLU, GroupNorm—and optimizes tensor layouts for Jetson Tensor Cores)._

---

## 📊 Step 3: Run the Benchmark & Compare Models

Run the benchmark using the compiled TensorRT engine:

```bash
python scripts/evaluate_mse.py \
  --checkpoint diamond_highway_mcts.pt \
  --trt_engine diamond_unet_fp16.engine \
  --dataset_path dataset_mcts \
  --samples 500 \
  --batch_size 16 \
  --steps 2
```

### Example Benchmark Output:

```text
============================================================
 DIAMOND Edge Evaluation Benchmark (MSE & Latency)
============================================================
 Device:          cuda:0
 Batch Size:      16
 FP16 Mixed Prec: True
 TRT Engine:      diamond_unet_fp16.engine
 Diffusion Steps: 2 (Overridden via CLI)

Evaluating: 100%|██████████████████████████████████| 500/500

============================================================
 RESULTS SUMMARY
============================================================
 Evaluated Samples:       500
 Final Pure MSE Score:    0.004218
 Average Latency / Frame: 42.15 ms
 Throughput:              23.72 frames/sec (FPS)
============================================================
```

---

## ⚙️ CLI Reference for `scripts/evaluate_mse.py`

| Argument         | Type   | Default        | Description                                           |
| :--------------- | :----- | :------------- | :---------------------------------------------------- |
| `--checkpoint`   | `str`  | _(Required)_   | Path to model weights (`diamond_highway_mcts.pt`)     |
| `--trt_engine`   | `str`  | `None`         | Path to compiled TensorRT `.engine` file              |
| `--dataset_path` | `str`  | `dataset_mcts` | Path to test dataset directory                        |
| `--samples`      | `int`  | `500`          | Number of test segments to evaluate                   |
| `--batch_size`   | `int`  | `32`           | Parallel batch size ($B=16$ or $B=32$ recommended)    |
| `--steps`        | `int`  | `None` (3)     | Denoising diffusion steps override (`1`, `2`, or `3`) |
| `--num_workers`  | `int`  | `4`            | DataLoader background workers                         |
| `--no_fp16`      | `flag` | `False`        | Disable FP16 mixed precision                          |
| `--compile`      | `flag` | `False`        | Enable `torch.compile` (PyTorch 2.0+)                 |

---

## 🔬 Recommended Evaluation Matrix for Model Comparison

To benchmark DIAMOND against other world models / predictors on edge, evaluate across these three standardized operating points:

```bash
# 1. High-Fidelity Baseline (3 Steps, TensorRT FP16)
python scripts/evaluate_mse.py --checkpoint diamond_highway_mcts.pt --trt_engine diamond_unet_fp16.engine --batch_size 16 --steps 3

# 2. Balanced Edge (2 Steps, TensorRT FP16) - RECOMMENDED
python scripts/evaluate_mse.py --checkpoint diamond_highway_mcts.pt --trt_engine diamond_unet_fp16.engine --batch_size 16 --steps 2

# 3. Ultra-Low Latency (1 Step, TensorRT FP16)
python scripts/evaluate_mse.py --checkpoint diamond_highway_mcts.pt --trt_engine diamond_unet_fp16.engine --batch_size 16 --steps 1
```

---

## 🔧 Architecture & Integration Details

- **[`scripts/export_onnx.py`](file:///home/doa/projects/kinemamba/diamond/scripts/export_onnx.py)**: Extracts and exports `InnerModel` from `Agent.denoiser`.
- **[`scripts/build_trt_engine.sh`](file:///home/doa/projects/kinemamba/diamond/scripts/build_trt_engine.sh)**: Invokes `trtexec` with dynamic batch profiles ($B \in [1, 64]$) and FP16 optimizations.
- **[`src/models/diffusion/trt_inner_model.py`](file:///home/doa/projects/kinemamba/diamond/src/models/diffusion/trt_inner_model.py)**: Zero-copy PyTorch CUDA tensor integration that plugs directly into DIAMOND's `Denoiser` and `DiffusionSampler`.
- **[`scripts/evaluate_mse.py`](file:///home/doa/projects/kinemamba/diamond/scripts/evaluate_mse.py)**: Batched, AMP-accelerated test harness measuring both image MSE and hardware latency/throughput.
