#!/bin/bash
#SBATCH --job-name=diamond_pixel_cv
#SBATCH --output=/blue/iruchkin/khek.do/diamond_pixel_results/logs/pixel_cv_%A_%a.out
#SBATCH --error=/blue/iruchkin/khek.do/diamond_pixel_results/logs/pixel_cv_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
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
    --output /blue/iruchkin/khek.do/diamond_dataset_1000_200steps \
    --seed 0

# Python reads the fold index from Slurm automatically and aggregates completed folds.
# To resume, add --resume to this command and keep the same --output directory.
# For a new experiment, change --output instead.
python -u scripts/cross_validate_pixels.py \
    --data /blue/iruchkin/khek.do/diamond_dataset_1000_200steps \
    --output /blue/iruchkin/khek.do/diamond_pixel_results/cv_vehicle_weighted_pixel_200steps \
    --epochs 50 \
    --workers 8 \
    --pixel-mask-gain 20 \
    --pixel-mae-weight 1.0 \
    --batch-size 32
