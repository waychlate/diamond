# CLAUDE.md

## Purpose: DIAMOND is a baseline

This repo is a fork of DIAMOND (a diffusion world model), adapted to highway driving. It serves as a **baseline** for KineMamba (https://github.com/AnirudhhVenkat/kinemamba), a Mamba2 latent world model.

The research question is how KineMamba compares to DIAMOND at predicting the next frame's pixels and time-to-collision (TTC), and where KineMamba wins. For example, it may be slightly less accurate but much faster at inference. The goal is to show that Mamba is viable for edge autonomous driving.

DIAMOND is the reference point, not the product. A change here earns its place by making the comparison with KineMamba fairer, not by making DIAMOND as strong as possible.

## Fair comparison

Match KineMamba wherever a setting means the same thing in both models:

- **Data:** the shared 200-step highway dataset (`dataset_episodes_1000_200steps`).
- **Folds:** 5-fold CV from KineMamba's `src/make_cv_folds.py` (seed 0, sklearn `KFold` over the train and test episodes pooled). Both models read the same fold list.
- **Vehicle mask:** `src/vehicle_mask.py` is a verbatim copy of KineMamba's `src/mask_utils.py` at commit 0c7566d. Keep it identical to the original.
- **Vehicle weighting:** per-pixel weights `1 + gain · mask`, with gain 20.
- **Context:** 20 frames. DIAMOND conditions on all 20 (`num_steps_conditioning: 20`, with `cond_channels: 240` so the action embedding divides evenly). Both models are scored on targets from frame 20 onward.
- **Training loss:** the vehicle-weighted pixel MAE is the only loss, and TTC is trained with MAE too, as in KineMamba.
- **Headline metric:** pixel MAE, both unmasked and masked (see `CONTEXT.md`), on next-frame prediction, with TTC MAE alongside.
- **Pixel scale:** KineMamba normalizes pixels to [-0.5, 0.5] (`x/255 - 0.5`), while DIAMOND uses [-1, 1]. The MAE formulas are the same, so the same image error gives DIAMOND an MAE twice as large. Halve DIAMOND's pixel MAE before putting it next to KineMamba's.

Some settings depend on the architecture: epochs, batch size, learning rate, optimizer, context handling. Record those instead of matching them, together with training wall time, GPU, parameter count, inference latency and peak memory.

When a change could affect fairness and these rules don't settle it, ask the user.

## Where things are

- **Vocabulary:** `CONTEXT.md` is the glossary. Use its terms.
- **Pixel CV workflow:** `docs/pixel_evaluation.md` describes the workflow, and `slurm/README.md` describes the jobs.
- **Training loss:** `src/models/diffusion/denoiser.py`.
- **TTC head:** `src/models/ttc_head.py` predicts TTC from the U-Net bottleneck latents and is trained separately (`scripts/train_latent_ttc.py`).
- **`viz/`:** local copies of cluster results, pulled with rsync. Treat it as data, not source.

## Running

- **Tests:** run `.venv/bin/python -m unittest discover tests`. pytest isn't installed.
- **Cluster jobs:** training and evaluation run on UF HiPerGator, which Claude can't reach. Hand the user commands to run and ask them to paste the output.
- **Cluster paths:** the checkout is `/home/khek.do/diamond-pixel-cv`, and results go to `/blue/iruchkin/khek.do/diamond_pixel_results/`. Local edits only take effect after the user syncs them to the cluster.

## Cluster gotchas

- **Group quota:** the `iruchkin` group's `/blue` quota (5 TB) fills up from other users' jobs. When it's full, jobs die with `Disk quota exceeded`, `PytorchStreamWriter failed writing file`, or a cut-off `object address : 0x…` dump with no traceback. Check `blue_quota` first.
- **Existing fold directory:** a fold whose output directory already exists stops with `FileExistsError`. Delete the dead fold directory, or pass `--resume` to continue from its checkpoint.
- **Resources:** the CV training job runs with 8 CPUs, 8 workers and 32 GB. Fold 0 peaked at about 37 GB and took about 1d 18h of the 48h limit.
- **Reading logs:** tqdm writes carriage returns, so read `.err` files with `tr '\r' '\n' < file | tail`.
