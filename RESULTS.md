# Evaluation Results

## Model: DIAMOND (Diffusion World Model)
* **Date**: June 10, 2026
* **Environment**: Highway-v0 (Custom MCTS Dataset)
* **Training Epochs**: 50
* **Total Steps**: 180,000

### Metrics
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Pure MSE** | **0.003497** | Evaluated on 500 test samples (scripts/evaluate_mse.py) |

### Artifacts
* **Checkpoint**: 'outputs/2026-06-05/20-31-42/checkpoints/state.pt'
* **Config**: 'outputs/2026-06-05/20-31-42/config/trainer.yaml