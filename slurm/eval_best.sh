#!/bin/bash
#SBATCH --job-name=diamond_eval_best
#SBATCH --output=/blue/iruchkin/khek.do/diamond_pixel_results/logs/eval_best_%j.out
#SBATCH --error=/blue/iruchkin/khek.do/diamond_pixel_results/logs/eval_best_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Throwaway: score best.pt of the gain-40 smoke run on fold 0's test split.
# An existing pixel_eval_best directory makes evaluate_pixels.py stop, so delete it to rerun.
set -euo pipefail

cd /home/khek.do/diamond-pixel-cv
module purge
module load python/3.11
module load cuda/12.1.1
source /home/khek.do/diamond/.venv/bin/activate

R=/blue/iruchkin/khek.do/diamond_pixel_results/cv_kinemamba_parity_gain40_smoke/fold_0

python -u scripts/evaluate_pixels.py \
    --dataset /blue/iruchkin/khek.do/diamond_dataset_1000_200steps_kfold/folds/fold_0/test \
    --checkpoint "$R/checkpoints/best.pt" \
    --config "$R/config/trainer.yaml" \
    --checkpoint-training-sigma 0.15 --sigma-data 0.15 \
    --output "$R/pixel_eval_best"
