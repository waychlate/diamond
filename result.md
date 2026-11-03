# Vehicle-weighted pixel loss vs. baseline — fold 0

> The tables from here to "Model cost" use the old protocol (old split, 4-frame context, MSE + MAE loss) and are superseded by the parity runs at the end.

## Setup

- Data: `diamond_dataset_1000_200steps`, fold 0 (1440 train / 360 test episodes)
- Eval: one-step prediction with real context (4 conditioning frames), seed 0, 70,560 test transitions
- Metric: MAE in [-1, 1] pixel space; masked MAE uses the soft vehicle mask (blue = 1, green = 0.6)
- Baseline: `pixel_mask_gain = 0`, `pixel_mae_weight = 0`
- Vehicle-weighted: `pixel_mask_gain = 20`, `pixel_mae_weight = 1.0`

## Results

|                                  | baseline        | vehicle-weighted | change |
| -------------------------------- | --------------- | ---------------- | ------ |
| unmasked MAE                     | 0.02297         | 0.01849          | −19.5% |
| masked (vehicle) MAE             | 0.05937         | 0.05753          | −3.1%  |
| masked MAE, episode mean ± std   | 0.0585 ± 0.0105 | 0.0567 ± 0.0102  |        |
| unmasked MAE, episode mean ± std | 0.0230 ± 0.0040 | 0.0185 ± 0.0040  |        |

## Paired per-episode comparison

Both models were evaluated on the same 360 test episodes, so differences are paired by episode.

|              | mean diff (weighted − baseline) | std of diff | paired t | weighted better in |
| ------------ | ------------------------------- | ----------- | -------- | ------------------ |
| unmasked MAE | −0.00448                        | 0.00178     | −47.8    | 359 / 360 episodes |
| masked MAE   | −0.00180                        | 0.00720     | −4.8     | 216 / 360 episodes |

## Takeaways

- The weighted loss lowers whole-frame error by about 20%, and does so in nearly every episode.
- Vehicle-region error drops by about 3%. The gain is small but consistent in the paired test, although the weighted model wins only 60% of episodes.
- Caveats: this is one fold with one training seed per model, and episodes are not fully independent, so the t-values overstate confidence. Folds 1–4 are needed before drawing conclusions.

## Source files

- `viz/cv_baseline_pixel_200steps/fold_0/pixel_eval/`
- `viz/cv_vehicle_weighted_pixel_200steps/fold_0/pixel_eval/`

Each folder contains `summary.json`, `episodes.csv`, `transitions.csv` and `example_00.png`–`example_05.png`.

## Model cost — vehicle-weighted, fold 0, epoch 50

Checkpoint: `cv_vehicle_weighted_pixel_200steps/fold_0/checkpoints/agent_versions/agent_epoch_00050.pt`, the second-to-last saved version of the old-protocol run. It uses 4 conditioning frames (`cond_channels` 256), not the 20 of the parity protocol.

Measured with `scripts/model_cost.py`: batch size 1, random 48×320 input, Euler sampler with 3 denoising steps, 20 warmup calls and then 200 timed calls. Latency covers one full next-frame prediction (all 3 denoiser passes).

| component            | parameters | used for next-frame pixels |
| -------------------- | ---------- | -------------------------- |
| denoiser (U-Net)     | 4,406,019 (16.8 MB fp32) | yes          |
| reward/end model     | 17,435,328 | no                         |
| actor-critic         | 8,997,318  | no                         |
| total in checkpoint  | 30,838,665 |                            |

| metric (RTX 4070 Laptop GPU) | value    |
| ---------------------------- | -------- |
| latency, mean                | 33.9 ms  |
| latency, median              | 31.4 ms  |
| latency, p95                 | 47.4 ms  |
| peak GPU memory              | 197 MB   |

Notes:

- Parameters are counted as `sum(p.numel())` and the size as params × 4 bytes, the same as KineMamba's `check_mamba_size.py`. That script counts only the Mamba2 dynamics core. For a like-for-like number, KineMamba's count should include its encoder, decoder and TTC head, because DIAMOND's U-Net does all of that work.
- Compare KineMamba against the denoiser's 4.4M parameters. The reward/end model and the actor-critic are DIAMOND's RL parts and play no part in predicting pixels. The TTC head isn't in this checkpoint, so it isn't counted here.
- The latency was measured on a laptop GPU. For the paper, rerun on the same GPU as KineMamba's timing.
- The 20-frame parity model only widens `conv_in` and the action embedding, so its parameter count barely changes. Measure its latency separately once its checkpoint exists.

## Parameters used for the next frame: DIAMOND vs KineMamba

These are the parameters that run when predicting the next frame and its TTC. They use KineMamba's labels from `check_sizes.py`, and MB is params × 4 bytes (fp32).

| part          | DIAMOND                   | KineMamba                  |
| ------------- | ------------------------- | -------------------------- |
| Encoder       | 1,236,672 (4.7 MB)        | 8,124,672 (31.0 MB)        |
| World_Model   | 3,169,347 (12.1 MB)       | 65,629,539 (250.4 MB)      |
| TTC_Predictor | 492,545 (1.9 MB)          | 295,425 (1.1 MB)           |
| **Total**     | **4,898,564 (18.7 MB)**   | **74,049,636 (282.5 MB)**  |

How DIAMOND's U-Net denoiser maps onto the labels:

- **Encoder:** `conv_in`, the down blocks and the downsamples. Its input is the context frames plus the noisy target, stacked as channels.
- **World_Model:** the action and noise conditioning, the bottleneck blocks, the up blocks with skip connections, and the output conv. It plays the role of KineMamba's dynamics plus decoder. Unlike KineMamba, DIAMOND has no separate latent dynamics module, and the action conditioning also reaches the down blocks.
- **TTC_Predictor:** `LatentTTCHead` without the LSTM (hidden 128, 4×4 pool), as planned for the parity TTC head. It isn't trained yet, so the count comes from the architecture. The old LSTM version had 1,345,409 parameters.

The denoiser counts come from the 4-frame epoch-50 checkpoint. The 20-frame model has a wider `conv_in` and `cond_channels` 240 instead of 256, so both parts will change slightly. Rerun the script on its checkpoint. KineMamba's counts are his `check_sizes.py` output, and they match a hand count from his source.

The reward/end model and the actor-critic are left out: neither is used to predict pixels, and the actor-critic is never trained (`start_after_epochs: 999999`).

## Parity smoke runs — fold 0, 20-frame context, MAE-only loss

These are the first runs under the KineMamba parity protocol, so they are not comparable with the tables above (old split, 4-frame context, MSE + MAE loss).

Setup, shared by both runs:

- Data: KineMamba's pooled folds (`cv_folds.json`, seed 0), fold 0. Test split: 400 episodes, 72,000 transitions (targets from frame 20 on).
- Model: `num_steps_conditioning` 20, `cond_channels` 240, 4,302,815 denoiser parameters.
- Loss: vehicle-weighted pixel MAE only (`diffusion_mse_weight` 0). 25 epochs, seed 0, `sigma_data` 0.15.
- Eval: one-step prediction with real context, Euler sampler with 3 denoising steps, final checkpoint (`state.pt`).
- Differences between the runs: the mask gain, and the reward/end model (trained in the gain-20 run, switched off in the gain-40 run, which does not affect the pixel loss but changes the random streams).

| run | unmasked MAE (KineMamba scale) | masked MAE (KineMamba scale) | unmasked MAE (DIAMOND scale) | masked MAE (DIAMOND scale) | masked, episode mean ± std (KineMamba scale) |
| --- | --- | --- | --- | --- | --- |
| gain 20 | 0.01393 | 0.03179 | 0.02786 | 0.06358 | 0.0312 ± 0.0055 |
| gain 40 | 0.01161 | 0.02592 | 0.02322 | 0.05185 | 0.0256 ± 0.0038 |

The KineMamba scale is the DIAMOND value divided by 2, because KineMamba's pixels are in [-0.5, 0.5] and DIAMOND's are in [-1, 1].

### Is gain 40 better than gain 20?

Gain 40 is the matched setting: KineMamba uses a flat 40, so the gain-40 run is the parity run and the gain-20 run is a side comparison. Whether 40 actually beats 20 for DIAMOND is not established. Gain 40 is 16.7% lower unmasked and 18.4% lower masked, but each is a single run. The gain-20 run's own checkpoints vary by that much from epoch to epoch:

| epoch (gain 20) | unmasked | masked |
| --- | --- | --- |
| 15 | 0.01179 | 0.02633 |
| 16 | 0.01244 | 0.03522 |
| 17 | 0.01298 | 0.02882 |
| 18 | 0.01233 | 0.02749 |
| 19 | 0.01330 | 0.03000 |
| 20 | 0.01285 | 0.02544 |
| 21 | 0.01348 | 0.02713 |
| 22 | 0.01346 | 0.03245 |
| 23 | 0.01454 | 0.02646 |
| 24 | 0.01349 | 0.03268 |
| 25 | 0.01393 | 0.03179 |

Gain 20 ranged over 0.01179–0.01454 unmasked and 0.02544–0.03522 masked across epochs 15–25. The gain-40 result (0.01161 / 0.02592) sits at the best end of that range. A second seed, or a gain-40 epoch curve, is needed to separate a real effect from run-to-run noise. Epochs 1–14 of the gain-20 run were deleted by `num_to_keep: 11`, so its early trend is not recoverable.

### Checkpoint selection

- `best.pt` (lowest mean held-out denoiser test loss) was epoch 25 for gain 40, the last epoch, with test loss 0.00189.
- Evaluating `best.pt` and the final `state.pt` gives identical results (all 400 per-episode values match), so the evaluation is deterministic and for this run best equals last.

### Not yet recorded

- Training wall time and GPU for both runs (`sacct -j <jobid> --format=Elapsed,MaxRSS`). From the log speeds, the gain-20 run took roughly 36 h, about half of each epoch going to the reward/end model.
- Gain-40 epoch curve, TTC MAE, and the other four folds.
