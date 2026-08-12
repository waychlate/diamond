#!/bin/bash
#SBATCH --job-name=diamond_ttc_eval
#SBATCH --output=logs/diamond_ttc_%j.log
#SBATCH --error=logs/diamond_ttc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

echo "Job Start"
date;hostname;pwd
echo "---"

module purge
module load python/3.11
module load cuda/12.1.1

cd /home/khek.do/diamond

source .venv/bin/activate

# Execute DIAMOND visual rollout + TTC Trajectory evaluation
python scripts/evaluate_diamond_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --ttc-model /home/khek.do/TTC-Prediction/best_model.pth \
    --dataset_path dataset_mcts \
    --episodes 10 \
    --rollout_steps 30 \
    --output_dir visualizations/diamond_ttc_rollouts

echo "Job End"
date
echo "---"
