#!/usr/bin/env bash

#SBATCH -p gpu22
#SBATCH -t 0-01:00:00
#SBATCH --gres gpu:2
#SBATCH -o "<absolute-path-to-code>/slurmlogs/%A-%a.out"

echo "$SLURM_JOB_ID" > "$SLURM_JOB_ID"

eval "$(conda shell.bash hook)"

conda activate <path-to-conda-env>

echo "Hello World"

nvidia-smi

cfg="configs/real/r.txt"

python ddp_test_nerf.py --config "$cfg" --render_split train --testskip 10
python ddp_test_nerf.py --config "$cfg" --render_split drunk2 --testskip 10
python ddp_test_nerf.py --config "$cfg" --render_split drunk1 --testskip 10
python ddp_test_nerf.py --config "$cfg" --render_split drunk --testskip 10

echo Finished
