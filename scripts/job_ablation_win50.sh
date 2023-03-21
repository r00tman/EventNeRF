#!/usr/bin/env bash

#SBATCH -p gpu20
#SBATCH -t 0-06:00:00
#SBATCH --gres gpu:2
#SBATCH -a 1-5%1
#SBATCH -o "<absolute-path-to-code>/slurmlogs/%A-%a.out"

echo "$SLURM_JOB_ID" > "$SLURM_JOB_ID"

eval "$(conda shell.bash hook)"

conda activate <path-to-conda-env>

echo "Hello World"

nvidia-smi

python ddp_train_nerf.py --config configs/ablation/win50.txt

echo Finished
