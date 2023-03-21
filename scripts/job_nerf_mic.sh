#!/usr/bin/env bash

#SBATCH -p gpu22
#SBATCH -t 0-06:00:00
#SBATCH --gres gpu:2
#SBATCH -a 1-5%1
#SBATCH -o "<absolute-path-to-code>/slurmlogs/%A-%a.out"

echo "$SLURM_JOB_ID" > "$SLURM_JOB_ID"

eval "$(conda shell.bash hook)"

conda activate <path-to-conda-env>

echo "Hello World"

nvidia-smi

python ddp_train_nerf.py --config configs/nerf/mic.txt

echo Finished
