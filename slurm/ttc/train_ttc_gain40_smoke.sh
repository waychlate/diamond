#!/bin/bash
#SBATCH --job-name=diamond_ttc_smoke
#SBATCH --output=/blue/iruchkin/khek.do/diamond_pixel_results/logs/ttc_smoke_%j.out
#SBATCH --error=/blue/iruchkin/khek.do/diamond_pixel_results/logs/ttc_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Train the TTC head on the frozen best.pt of the gain-40 smoke run (fold 0), then score the whole test split.
# Parity with KineMamba: next-frame target, L1 loss, no LSTM. The smoke model was trained on the cropped
# dataset, so this reads that fold, not the _full one used by the 5-fold CV.
# Outputs go to a new ttc_head/ directory; delete it to rerun.
set -euo pipefail

cd /home/khek.do/diamond-pixel-cv
module purge
module load python/3.11
module load cuda/12.1.1
source /home/khek.do/diamond/.venv/bin/activate

R=/blue/iruchkin/khek.do/diamond_pixel_results/cv_kinemamba_parity_gain40_smoke/fold_0

python -u scripts/train_latent_ttc.py \
    --checkpoint "$R/checkpoints/best.pt" \
    --config "$R/config/trainer.yaml" \
    --dataset_path /blue/iruchkin/khek.do/diamond_dataset_1000_200steps_kfold/folds/fold_0 \
    --save_path "$R/ttc_head/best_ttc.pt" \
    --eval_output "$R/ttc_head/eval" \
    --target_offset 1 \
    --loss_fn l1 \
    --epochs 20 \
    --steps_per_epoch 500 \
    --batch_size 32 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --dropout 0.1 \
    --num_workers 4 \
    --seed 0 \
    --device cuda
