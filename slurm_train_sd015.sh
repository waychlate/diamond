#!/bin/bash
#SBATCH --job-name=diamond_sd015
#SBATCH --output=logs/diamond_sd015_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
set -e
mkdir -p logs
module purge; module load python/3.11 cuda/12.1.1
cd /home/khek.do/diamond
source .venv/bin/activate
# Only sigma_data (and seed) differ from the 0.5 baseline. To resume after timeout:
# python src/main.py common.resume=True hydra.output_subdir=null hydra.run.dir=.   (from the run's output dir)
python src/main.py agent.denoiser.sigma_data=0.15 common.seed=0 \
    static_dataset.path=/blue/iruchkin/khek.do/diamond_dataset_1000 \
    training.num_workers_data_loaders=8 training.cache_in_ram=False
