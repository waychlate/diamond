#!/bin/bash
#SBATCH --job-name=diamond_latent_ttc_eval
#SBATCH --output=logs/latent_ttc_%j.log
#SBATCH --error=logs/latent_ttc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

echo "Job Start"
date; hostname; pwd
echo "---"

set -e

mkdir -p logs
mkdir -p checkpoints
mkdir -p visualizations/latent_ttc_eval_50eps

module purge
module load python/3.11
module load cuda/12.1.1

cd /home/khek.do/diamond

source .venv/bin/activate

# Evaluate Latent TTC Model across rollout horizons (50 episodes, 20-frame context, +30 lookahead steps)
echo "--- Running Latent TTC Evaluation (50 Episodes) ---"
python scripts/evaluate_diamond_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --ttc-model checkpoints/best_latent_ttc.pt \
    --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000 \
    --mode latent \
    --episodes 50 \
    --context_len 20 \
    --rollout_steps 30 \
    --dt 0.1 \
    --output_dir visualizations/latent_ttc_eval_50eps

echo "Job End"
date
echo "---"
