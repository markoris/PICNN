#!/bin/sh
#SBATCH --job-name ray_tune
#SBATCH --account=t25_mlgpu_g
#SBATCH --time=10:00:00
#SBATCH --nodes=4
#SBATCH -p gpu

module load python PrgEnv-nvhpc

python -u ray_tune_optim.py
