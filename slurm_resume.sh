#!/bin/bash
#SBATCH --job-name=diamond_resume
#SBATCH --output=resume_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
# Resume a training run from its latest checkpoints/state.pt.
# Usage:  cd <run folder>  &&  sbatch /home/khek.do/diamond/slurm_resume.sh [extra hydra overrides]
# The run folder's own config/ and src/ are used, so sigma_data, seed, dataset etc. are exactly as in the original run.
# The log is written to resume_<jobid>.log in the run folder.
set -e
RUN_DIR=$SLURM_SUBMIT_DIR
cd "$RUN_DIR"
[ -f checkpoints/state.pt ] || { echo "No checkpoints/state.pt in $RUN_DIR. cd into the run folder before sbatch."; exit 1; }
[ ! -f .run_is_over ] || { echo "Run is marked finished ($RUN_DIR/.run_is_over). Delete that file to resume anyway."; exit 1; }
echo "Resuming $RUN_DIR"; cat checkpoints/info_for_import_script.json 2>/dev/null; echo; date; hostname
grep -m1 "sigma_data" config/trainer.yaml
module purge; module load python/3.11 cuda/12.1.1
source /home/khek.do/diamond/.venv/bin/activate
python -u src/main.py common.resume=True hydra.output_subdir=null hydra.run.dir=. "$@"
echo "Job End"; date
