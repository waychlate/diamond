#!/bin/bash
#SBATCH --job-name=mcts_episode_gen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/diamond_%j.log
echo "Job Start"
date;hostname;pwd
echo "---"

module purge
module load python/3.11
module load cuda/12.1.1

cd /home/khek.do/diamond

python -m venv .venv

source .venv/bin/activate

python -u src/main.py     # Python uses its own internal pooling to fill the 10 cores

echo "Job End"
date
echo "---"