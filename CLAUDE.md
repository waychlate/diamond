# CLAUDE.md

## Purpose: DIAMOND is a baseline

This repo is a fork of DIAMOND (a diffusion world model), adapted to highway driving. It serves as a **baseline** for KineMamba (https://github.com/AnirudhhVenkat/kinemamba), a Mamba2 latent world model.

The research question is how KineMamba compares to DIAMOND at predicting the next frame's pixels and time-to-collision (TTC), and where KineMamba wins. For example, it may be slightly less accurate but much faster at inference. The goal is to show that Mamba is viable for edge autonomous driving.

DIAMOND is the reference point, not the product. A change here earns its place by making the comparison with KineMamba fairer, not by making DIAMOND as strong as possible.

## Fair comparison

Match KineMamba wherever a setting means the same thing in both models:

- **Data:** the shared 200-step highway dataset (`dataset_episodes_1000_200steps`).
- **Images:** the whole 150×600 frame resized to 48×320 (bicubic, antialiased, rounded to uint8) as in KineMamba's loader, via `prepare_pixel_data.py --no-crop`. The older crop (rows 60–150, bilinear) is the default only so the pretrained-checkpoint evaluation keeps working. The two parity smoke runs used the crop, so their pixel numbers are not comparable with KineMamba's.
- **Folds:** 5-fold CV from KineMamba's `src/make_cv_folds.py` (seed 0, sklearn `KFold` over the train and test episodes pooled). Both models read the same fold list.
- **Vehicle mask:** `src/vehicle_mask.py` is a verbatim copy of KineMamba's `src/mask_utils.py` at commit 0c7566d. Keep it identical to the original.
- **Vehicle weighting:** per-pixel weights `1 + gain · mask`, with gain 40 (KineMamba uses a flat 40, confirmed by its author). The earlier gain-20 smoke run is a side comparison only.
- **Context:** 20 frames. DIAMOND conditions on all 20 (`num_steps_conditioning: 20`, with `cond_channels: 240` so the action embedding divides evenly). Both models are scored on targets from frame 20 onward.
- **Training loss:** the vehicle-weighted pixel MAE is the only loss, and TTC is trained with MAE too, as in KineMamba.
- **Headline metric:** pixel MAE, both unmasked and masked (see `CONTEXT.md`), on next-frame prediction, with TTC MAE alongside.
- **Pixel scale:** KineMamba normalizes pixels to [-0.5, 0.5] (`x/255 - 0.5`), while DIAMOND uses [-1, 1]. The MAE formulas are the same, so the same image error gives DIAMOND an MAE twice as large. Halve DIAMOND's pixel MAE before putting it next to KineMamba's.
- **Unmasked means unweighted:** the unmasked pixel MAE is the plain MAE over all pixels, not the gain-weighted training loss. At gain 40 the weighted loss reads about 23% higher than the plain MAE on the same predictions.
- **Aggregation:** pool before dividing (sum of absolute errors over sum of pixels or mask weights), not a mean of per-batch ratios.
- **Checkpoint selection:** both models keep a `best.pt` chosen on the held-out fold that is also reported (KineMamba by validation pixel loss + TTC loss; DIAMOND by mean denoiser test loss, or by weighted pixel MAE + TTC MAE with weight 1 when the TTC head is trained jointly, to mirror KineMamba). Say so when reporting best-epoch numbers.

- **Parameter count:** the plan is to meet KineMamba in the middle. DIAMOND grows through a KineMamba-style dense bottleneck (`--dense-bottleneck-dim 1024`: flatten the bottleneck map, Linear →1024, SiLU, Linear 1024→map, added back), and the TTC head reads the 1024-d vector instead of the pooled map. See `docs/parameter_scaling.md` (option 1.2). The current plan adds `--kinemamba-unet` (`KineMambaUNet` in `src/models/blocks.py`): KineMamba's encoder/decoder layout (widths [128,192,256,256], 4 downsamples to 256×3×20, his additive 1×1 skips), noise conditioning in every ResBlock, actions only at the dense bottleneck (32 per step, LayerNorm, as his `proj_in`). Total 55.09M with the TTC head, 39.3 GMACs per denoising pass (3.3× the old 11.9). Without the flag: the old `[64]*4` U-Net, 36.07M.

Some settings depend on the architecture: epochs, batch size, learning rate, optimizer, context handling. Record those instead of matching them, together with training wall time, GPU, parameter count, inference latency and peak memory.

Hardware: KineMamba trains on the `hpg-b200` and `hpg-turin` partitions. Training wall time depends on the GPU, so record which one each run used. The headline latency comparison will run on a Jetson Orin Nano, so GPU class doesn't need to match for training. If `hpg-b200` is faster for DIAMOND, using it is fine.

When a change could affect fairness and these rules don't settle it, ask the user.

## Where things are

- **Vocabulary:** `CONTEXT.md` is the glossary. Use its terms.
- **Pixel CV workflow:** `docs/pixel_evaluation.md` describes the workflow, and `slurm/README.md` describes the jobs.
- **Training loss:** `src/models/diffusion/denoiser.py`.
- **TTC head:** `src/models/ttc_head.py` is KineMamba's TTCPredictor MLP (1024→256→128→1, ReLU, dropout 0.2, 295,425 params), fed the 64×6×40 bottleneck average-pooled to 64×2×8 = 1024 features. Heads trained before this change (conv + MLP, 492,545 params, including the 0.790 s smoke result) cannot be loaded with it. Two ways to train it: separately on a frozen denoiser (`scripts/train_latent_ttc.py`), or jointly with the denoiser (`--ttc-loss-weight` > 0, see `ttc_loss_weight` in `src/models/diffusion/denoiser.py`), as KineMamba does; the joint head is scored by `scripts/evaluate_ttc.py` into `ttc_eval/`.
- **`viz/`:** local copies of cluster results, pulled with rsync. Treat it as data, not source.

## Running

- **Tests:** run `.venv/bin/python -m unittest discover tests`. pytest isn't installed.
- **Cluster jobs:** training and evaluation run on UF HiPerGator, which Claude can't reach. Hand the user commands to run and ask them to paste the output.
- **Cluster paths:** the checkout is `/home/khek.do/diamond-pixel-cv`, and results go to `/blue/iruchkin/khek.do/diamond_pixel_results/`. Local edits only take effect after the user syncs them to the cluster.

## Cluster gotchas

- **Group quota:** the `iruchkin` group's `/blue` quota (5 TB) fills up from other users' jobs. When it's full, jobs die with `Disk quota exceeded`, `PytorchStreamWriter failed writing file`, or a cut-off `object address : 0x…` dump with no traceback. Check `blue_quota` first.
- **Existing fold directory:** a fold whose output directory already exists stops with `FileExistsError`. Delete the dead fold directory, or pass `--resume` to continue from its checkpoint.
- **Resources:** the CV training job runs with 8 CPUs, 8 workers and 64 GB. Fold 0 peaked at about 37 GB and took about 1d 18h of the 48h limit.
- **Reading logs:** tqdm writes carriage returns, so read `.err` files with `tr '\r' '\n' < file | tail`.
