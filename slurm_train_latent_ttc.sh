#!/bin/bash
#SBATCH --job-name=diamond_latent_ttc
#SBATCH --output=logs/latent_ttc_%j.log
#SBATCH --error=logs/latent_ttc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

echo "Job Start"
date; hostname; pwd
echo "---"

set -e

mkdir -p logs
mkdir -p checkpoints
mkdir -p visualizations/latent_ttc_eval

module purge
module load python/3.11
module load cuda/12.1.1

cd /home/khek.do/diamond

source .venv/bin/activate

# 1. (Optional) Convert raw dataset if not already converted
# python scripts/convert_and_process.py \
#     --src_dir /blue/iruchkin/khek.do/dataset_episodes_1000 \
#     --dst_dir /blue/iruchkin/khek.do/diamond_dataset_mcts

# 1. Train Latent TTC Head directly from DIAMOND UNet bottleneck features (UNCAPPED)
echo "--- Starting Latent TTC Head Training ---"
python scripts/train_latent_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000 \
    --save_path checkpoints/best_latent_ttc.pt \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --context_len 20 \
    --dt 0.1 \
    --dropout 0.1 \
    --device cuda

# 2. Evaluate Latent TTC Model across rollout horizons (UNCAPPED)
echo "--- Running Latent TTC Evaluation ---"
python scripts/evaluate_diamond_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --ttc-model checkpoints/best_latent_ttc.pt \
    --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000 \
    --mode latent \
    --episodes 10 \
    --context_len 20 \
    --rollout_steps 30 \
    --dt 0.1 \
    --output_dir visualizations/latent_ttc_eval

echo "Job End"
date
echo "---"

