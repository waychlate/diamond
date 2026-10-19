#!/bin/bash
#SBATCH --job-name=diamond_sd
#SBATCH --output=logs/diamond_sd_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
# Usage: sbatch slurm_train_sd.sh 0.15   (and 0.5 for the control)
set -e
SIGMA=${1:?usage: sbatch slurm_train_sd.sh <sigma_data>}
echo "Job Start  SIGMA=$SIGMA"; date; hostname
mkdir -p logs
module purge; module load python/3.11 cuda/12.1.1
cd /home/khek.do/diamond
source .venv/bin/activate
# Only sigma_data differs between runs. To resume after timeout, from the run's output dir:
# python src/main.py common.resume=True hydra.output_subdir=null hydra.run.dir=.
python -u src/main.py agent.denoiser.sigma_data=$SIGMA common.seed=0 wandb.name=sd$SIGMA hydra.run.dir=outputs/sd${SIGMA}_${SLURM_JOB_ID} \
    static_dataset.path=/blue/iruchkin/khek.do/diamond_dataset_1000 \
    training.num_workers_data_loaders=8 training.cache_in_ram=False
echo "Job End"; date
