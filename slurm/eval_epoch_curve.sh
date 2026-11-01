#!/bin/bash
#SBATCH --job-name=diamond_epoch_curve
#SBATCH --output=/blue/iruchkin/khek.do/diamond_pixel_results/logs/epoch_curve_%A_%a.out
#SBATCH --error=/blue/iruchkin/khek.do/diamond_pixel_results/logs/epoch_curve_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --array=1-25%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Throwaway: evaluate every saved epoch of the parity smoke run on fold 0's test split,
# to see whether MAE is still improving. One array task per epoch, 5 at a time.
# An existing output directory makes evaluate_pixels.py stop, so delete it to rerun an epoch.
set -euo pipefail

cd /home/khek.do/diamond-pixel-cv
module purge
module load python/3.11
module load cuda/12.1.1
source /home/khek.do/diamond/.venv/bin/activate

RUN=/blue/iruchkin/khek.do/diamond_pixel_results/cv_kinemamba_parity_smoke/fold_0
EPOCH=$(printf "%05d" "$SLURM_ARRAY_TASK_ID")

python -u scripts/evaluate_pixels.py \
    --dataset /blue/iruchkin/khek.do/diamond_dataset_1000_200steps_kfold/folds/fold_0/test \
    --checkpoint "$RUN/checkpoints/agent_versions/agent_epoch_$EPOCH.pt" \
    --config "$RUN/config/trainer.yaml" \
    --checkpoint-training-sigma 0.15 --sigma-data 0.15 \
    --output "$RUN/epoch_curve/epoch_$EPOCH"

# After all tasks finish, print the curve (masked and unmasked MAE, KineMamba scale):
#   for f in $RUN/epoch_curve/epoch_*/summary.json; do
#     python -c "import json,sys; s=json.load(open('$f')); print('$f'.split('/')[-2], s['unmasked_mae_kinemamba_scale'], s['masked_mae_kinemamba_scale'])"
#   done
