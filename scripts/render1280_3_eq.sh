#!/usr/bin/env bash

#SBATCH -p gpu20
#SBATCH -t 06:00:00
#SBATCH --gres gpu:2
#SBATCH -o "<absolute-path-to-code>/slurmlogs/%j.out"

eval "$(conda shell.bash hook)"

conda activate <path-to-conda-env>

echo "Hello World"

nvidia-smi

python ddp_test_nerf.py --config configs/lk2/c5_3_eq.txt \
                            --render_splits camera_path50

echo Finished
