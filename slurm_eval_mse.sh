#!/bin/bash
#SBATCH --job-name=diamond_eval_mse
#SBATCH --output=logs/eval_mse_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Usage: sbatch slurm_eval_mse.sh <path/to/agent_epoch_00050.pt> <sigma_data the model was trained with>
#   e.g. sbatch slurm_eval_mse.sh outputs/2026-10-19/17-35-11/checkpoints/agent_versions/agent_epoch_00050.pt 0.15
set -e
CKPT=${1:?usage: sbatch slurm_eval_mse.sh <checkpoint.pt> <sigma_data>}
SIGMA=${2:?usage: sbatch slurm_eval_mse.sh <checkpoint.pt> <sigma_data>}
mkdir -p logs
cd /home/khek.do/diamond
[ -f "$CKPT" ] || { echo "Checkpoint not found: $CKPT (path is relative to /home/khek.do/diamond)"; exit 1; }
echo "Eval $CKPT sigma_data=$SIGMA"; date; hostname
module purge; module load python/3.11 cuda/12.1.1
source .venv/bin/activate
python -u scripts/evaluate_mse.py --checkpoint "$CKPT" --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000 \
    --sigma_data "$SIGMA" --seed 0 --samples 500
echo "Job End"; date
