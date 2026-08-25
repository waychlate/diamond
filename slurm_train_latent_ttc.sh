#!/bin/bash
#SBATCH --job-name=diamond_latent_ttc
#SBATCH --output=logs/latent_ttc_%j.log
#SBATCH --error=logs/latent_ttc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -e  # Exit immediately if a command exits with a non-zero status

echo "Job Start"
date; hostname; pwd
echo "---"

mkdir -p logs
mkdir -p checkpoints
mkdir -p visualizations/latent_ttc_eval

module purge
module load python/3.11
module load cuda/12.1.1

cd /home/khek.do/diamond

source .venv/bin/activate

# 1. Train Latent TTC Head directly from DIAMOND UNet bottleneck features
echo "--- Starting Latent TTC Head Training ---"
python scripts/train_latent_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000 \
    --save_path checkpoints/best_latent_ttc.pt \
    --epochs 50 \
    --steps_per_epoch 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --context_len 20 \
    --dt 0.1 \
    --max_ttc 5.0 \
    --dropout 0.1 \
    --num_workers 0 \
    --device cuda

# 2. Evaluate Latent TTC Model across rollout horizons
echo "--- Running Latent TTC Evaluation ---"
python scripts/evaluate_diamond_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --ttc-model checkpoints/best_latent_ttc.pt \
    --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000 \
    --mode latent \
    --episodes 10 \
    --rollout_steps 30 \
    --dt 0.1 \
    --max_ttc 5.0 \
    --output_dir visualizations/latent_ttc_eval

echo "Job End"
date
echo "---"

