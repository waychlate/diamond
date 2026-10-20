# Slurm jobs

Submit from the repository root and create `logs/` before `sbatch` so Slurm can
open log files. Edit the literal paths directly in the two pixel jobs; no exports
or shared setup script are needed.

| File | Purpose |
| --- | --- |
| `eval_pixels.sh` | Existing-checkpoint masked/unmasked pixel MAE |
| `cross_validate_pixels.sh` | Five fresh DIAMOND training runs and pixel evaluation |
| `data/convert_dataset.sh` | Existing raw dataset conversion job |
| `ttc/train_latent_ttc.sh` | Existing latent TTC head training job |
| `ttc/eval_euler.sh` | Existing Euler TTC evaluation job |
| `ttc/eval_heun.sh` | Existing Heun TTC evaluation job |

See [pixel job instructions](../docs/pixel_evaluation.md) for dataset locations,
submission commands, outputs, and resuming CV tasks.

The existing jobs were moved from the root as follows. Their workload arguments
are preserved, including their older dataset paths. Their working directory now
uses `PROJECT_DIR` or `SLURM_SUBMIT_DIR` instead of a hardcoded checkout path.

| Previous path | Current path |
| --- | --- |
| `slurm_convert_dataset.sh` | `slurm/data/convert_dataset.sh` |
| `slurm_train_latent_ttc.sh` | `slurm/ttc/train_latent_ttc.sh` |
| `slurm_train_latent_ttc_eval.sh` | `slurm/ttc/eval_euler.sh` |
| `slurm_eval_heun.sh` | `slurm/ttc/eval_heun.sh` |
