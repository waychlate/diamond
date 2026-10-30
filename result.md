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
