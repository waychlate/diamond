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

# Execute DIAMOND visual rollout + TTC Lookahead evaluation with 20-frame context
python scripts/evaluate_diamond_ttc.py \
    --checkpoint diamond_highway_mcts.pt \
    --ttc-model /home/khek.do/diamond/best_model.pth \
    --dataset_path /blue/iruchkin/khek.do/dataset_episodes_1000 \
    --episodes 10 \
    --context_frames 20 \
    --lookahead_steps 30 \
    --output_dir visualizations/ttc_lookahead_eval

echo "Job End"
date
echo "---"
