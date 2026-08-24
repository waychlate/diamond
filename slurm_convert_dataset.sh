#!/bin/bash
#SBATCH --job-name=diamond_convert_data
#SBATCH --output=logs/convert_%j.log
#SBATCH --error=logs/convert_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

echo "Job Start: Converting Raw MCTS Episodes to DIAMOND Format"
date; hostname; pwd
echo "---"

mkdir -p logs

module purge
module load python/3.11

cd /home/khek.do/diamond

source .venv/bin/activate

# Paths:
# SRC_DIR: Directory containing raw 'train/' and 'test/' folders with episode_*.csv and episode_*.npz
# DST_DIR: Output directory where processed DIAMOND .pt dataset will be saved
SRC_DIR="/blue/iruchkin/khek.do/dataset_episodes_1000"
DST_DIR="/blue/iruchkin/khek.do/diamond_dataset_1000"

echo "Source raw dataset: $SRC_DIR"
echo "Destination DIAMOND dataset: $DST_DIR"

python scripts/convert_and_process.py \
    --src_dir "$SRC_DIR" \
    --dst_dir "$DST_DIR"

echo "Job End: Conversion Complete"
date
echo "---"

