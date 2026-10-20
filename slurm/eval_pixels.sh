#!/bin/bash
#SBATCH --job-name=diamond_pixel_eval
#SBATCH --output=logs/pixel_eval_%j.out
#SBATCH --error=logs/pixel_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

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

# Change --output for a new run; existing results are never overwritten.
python -u scripts/evaluate_pixels.py \
    --checkpoint /home/khek.do/diamond/diamond_highway_mcts.pt \
    --dataset /blue/iruchkin/khek.do/diamond_dataset_1000_200steps/test \
    --sigma-data 0.15 \
    --checkpoint-training-sigma 0.5 \
    --seed 0 \
    --order 1 \
    --batch-size 32 \
    --device cuda \
    --output /blue/iruchkin/khek.do/diamond_pixel_results/eval_pixel_200steps
