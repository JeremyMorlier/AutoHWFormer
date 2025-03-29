#!/bin/bash
#SBATCH --partition=Brain3080
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --output=log/%x/%j/logs.out
#SBATCH --error=log/%x/%j/errors.err

source .venv/bin/activate
srun python3 ZigZag_space.py