# Vehicle-weighted pixel loss vs. baseline — fold 0

Only fold 0 of the 5-fold CV has finished; folds 1–4 are pending.

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
