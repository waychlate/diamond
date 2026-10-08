# DIAMOND hyperparameters to try for TTC accuracy

Context: the TTC head is already accurate on real frames (about 0.6 to 0.9 s MAE, `real_control_mae_seconds`),
while Euler dream-frame MAE is about 5.1 s. Most of the error comes from world-model drift, so these levers
target the world model and the sampler. None of the expected effects below are tested.

Config files: `config/agent/default.yaml` (model, preconditioning) and `config/trainer.yaml` (training, sampler).

## Current values

| Parameter | Value | File |
|---|---|---|
| `sigma_data` | 0.5 | agent/default.yaml |
| `sigma_offset_noise` | 0.3 | agent/default.yaml |
| `num_steps_conditioning` | 4 | agent/default.yaml |
| `depths` / `channels` / `attn_depths` | [2,2,2,2] / [64,64,64,64] / [0,0,0,0] | agent/default.yaml |
| `num_autoregressive_steps` | 1 | trainer.yaml (denoiser.training) |
| `sigma_distribution` | loc -0.4, scale 1.2, range 2e-3 to 20 | trainer.yaml |
| `num_steps_denoising` | 3 | trainer.yaml (diffusion_sampler) |
| `sigma_min` / `sigma_max` / `rho` | 2e-3 / 5.0 / 7 | trainer.yaml |
| `order` | 1 (Euler) | trainer.yaml |
| `s_churn` | 0.0 | trainer.yaml |

Sampler sigmas for 3 steps: about [5, 0.27, 0.002, 0].

## Eval-only (no retraining)

1. **Fix the episode set.** The batch sampler picks random episodes each run, so runs are not comparable
   (the real-frame control moved from 0.91 s to 0.61 s between runs with the same head). Add a `--seed` to
   `scripts/evaluate_diamond_ttc.py` before comparing anything.
2. **`num_steps_denoising`.** Sweep 3, 4, 5, 8 with Euler. With `sigma_offset_noise: 0.3` the network never sees
   noise below about 0.3, so steps below that are wasted. Extra steps help between 0.3 and 5; `sigma_max` and
   `rho` control that spacing.
3. **Average TTC over several samples per step.** Cuts the variance of a single random sample. Costs N times the
   compute, so only use it in the baseline if the other models get the same treatment.

## Training (retrain the world model)

4. **`num_autoregressive_steps`.** With 1, the model only trains on real conditioning frames and never sees its own
   errors. Try 2 to 4 to train on its own predictions. Probably the biggest lever for drift. Check the training
   loop actually feeds the denoised outputs back in.
5. **Model capacity.** The U-Net is small and has no attention. Cars on a 150x600 road are small objects, so try
   wider channels (for example [64,128,128,256]) or attention at the low-resolution levels.
6. **Rare collision frames.** About 54% of frames are safe, and near-collision frames are a minority. Oversample
   them when training the denoiser to reduce phantom-car hallucinations. Pixel MSE hides small car-position errors
   (24 dB), and pixel-vs-TTC correlation was near zero, so track a car-position metric as well.
7. **`num_steps_conditioning`.** More context frames give better velocity and relative-speed estimates.
8. **Check convergence.** Look at the denoiser loss curve in wandb. If it is still falling, train longer first.

## Preconditioning (retrain)

9. **`sigma_data`.** EDM expects this to match the data's pixel standard deviation. **Measured: about 0.153**
   (mean -0.19, per-channel std 0.141 / 0.153 / 0.162, RMS 0.243), on 150 random train episodes, every 10th
   frame, with the same crop/resize/[-1,1] pipeline as `scripts/convert_and_process.py`. Per-episode std ranges
   0.143 to 0.164, so it's stable. The config value of 0.5 is over 3x too large. Try `sigma_data: 0.15`. If you
   change it, consider shifting `sigma_distribution.loc` by `ln(0.15/0.5)` (about -1.2, so about -1.6) to keep the
   noise levels relative to the data scale the same (a heuristic, untested).
10. **`sigma_offset_noise`.** Sets the noise floor the network sees. Lowering it (for example to 0.1) lets the
    model resolve fine detail at the end of sampling, but may bring back the brightness drift it was added to
    prevent.
11. **`sigma_distribution`.** Training sigmas are log-normal (median about 0.67). That covers sigma 5 and 0.27, but
    essentially never trains 0.002. The offset noise hides this.

## Suggested order

1 (fix the seed), then 2 (steps sweep), then 4 (autoregressive steps). Items 5 and 6 next if drift is still large.

## Heun findings (for reference)

Heun failed on this denoiser: dream PSNR fell from 24 dB to about 8 dB. Turning off 8-bit quantization in the
second denoise call did not fix it, so quantization was not the cause. The leading hypothesis is
`sigma_offset_noise: 0.3`: at sigma 0.002 the network is told the noise is about 0.3 and returns a smoothed
estimate, and Heun's correction then divides the residual by 0.002. This is untested. The
`quantize=False` experiment was reverted in the working tree (not committed yet), so `order: 2` is standard Heun.
