#!/bin/bash
#SBATCH --job-name=diamond_kmunet
#SBATCH --output=/blue/iruchkin/khek.do/diamond_pixel_results/logs/kmunet_cv_%A_%a.out
#SBATCH --error=/blue/iruchkin/khek.do/diamond_pixel_results/logs/kmunet_cv_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
# KineMamba-shaped U-Net (see CLAUDE.md). Fold 0 only; --array=0-4 for the full CV.
# Timing run first: sbatch --export=ALL,EPOCHS=1 <this script> (writes to a separate _timing_e1 directory).
# Resume an existing run: sbatch --export=ALL,RESUME=1 <this script>
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

# Before submitting: mkdir -p /blue/iruchkin/khek.do/diamond_pixel_results/logs
# Edit these paths to match your HiPerGator checkout and virtual environment.
cd /home/khek.do/diamond-pixel-cv
module purge
module load python/3.11
module load cuda/12.1.1
source /home/khek.do/diamond/.venv/bin/activate

python -u scripts/prepare_pixel_data.py \
    --source /blue/iruchkin/khek.do/dataset_episodes_1000_200steps \
    --output /blue/iruchkin/khek.do/diamond_dataset_1000_200steps_kfold_full \
    --seed 0 \
    --no-crop

# Python reads the fold index from Slurm automatically and aggregates completed folds.
# To resume, add --resume to this command and keep the same --output directory.
# For a new experiment, change --output instead.
python -u scripts/cross_validate_pixels.py \
    --data /blue/iruchkin/khek.do/diamond_dataset_1000_200steps_kfold_full \
    --output /blue/iruchkin/khek.do/diamond_pixel_results/cv_kinemamba_unet_gain40${EPOCHS:+_timing_e$EPOCHS} \
    --epochs ${EPOCHS:-25} \
    --workers 8 \
    --pixel-mask-gain 40 \
    --pixel-mae-weight 1.0 \
    --diffusion-mse-weight 0 \
    --batch-size 32 \
    --dense-bottleneck-dim 1024 \
    --kinemamba-unet ${RESUME:+--resume}
